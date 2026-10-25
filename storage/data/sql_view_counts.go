package data

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/gorse-io/gorse/storage"
	"github.com/pkg/errors"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// WithTwitchLiveProjection marks a source projection, not a general-purpose
// full replacement. Only this path may preserve independent label patches when
// the source's cold projection is unchanged.
type twitchLiveProjectionKey struct{}

func WithTwitchLiveProjection(ctx context.Context) context.Context {
	return context.WithValue(ctx, twitchLiveProjectionKey{}, true)
}

func IsTwitchLiveProjection(ctx context.Context) bool {
	projection, _ := ctx.Value(twitchLiveProjectionKey{}).(bool)
	return projection
}

// View counts stay numbers in the public labels object, but frequently changing
// Twitch live counts need not rewrite the embedding-bearing items row.
type sqlItemViewCount struct {
	ItemId         string `gorm:"column:item_id;primaryKey"`
	ViewCount      int64  `gorm:"column:view_count"`
	ProjectionHash []byte `gorm:"column:projection_hash"`
}

func (d *SQLDatabase) itemViewCountsTable() string { return string(d.TablePrefix) + "item_view_counts" }
func mysqlIdentifier(name string) string           { return "`" + strings.ReplaceAll(name, "`", "``") + "`" }

func (d *SQLDatabase) initItemViewCounts() error {
	return d.gormDB.Exec(fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		item_id VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL PRIMARY KEY,
		view_count BIGINT NOT NULL,
 projection_hash BINARY(32) NULL,
		FOREIGN KEY (item_id) REFERENCES %s (item_id) ON DELETE CASCADE
	) ENGINE=InnoDB`, mysqlIdentifier(d.itemViewCountsTable()), mysqlIdentifier(d.ItemsTable()))).Error
}

// A single statement reads metadata and the count from one committed snapshot.
// Legacy labels win during lazy conversion; untouched items need no backfill.
func (d *SQLDatabase) itemSelectColumns() string {
	if !d.separateTwitchViewCounts {
		return "item_id, is_hidden, categories, time_stamp, labels, comment"
	}
	base := mysqlIdentifier(d.ItemsTable())
	counts := mysqlIdentifier(d.itemViewCountsTable())
	labels := fmt.Sprintf(`CASE WHEN JSON_TYPE(%[1]s.labels) = 'OBJECT'
		AND NOT JSON_CONTAINS_PATH(%[1]s.labels, 'one', '$.viewCount')
		THEN COALESCE((SELECT JSON_SET(%[1]s.labels, '$.viewCount', counts.view_count)
		FROM %[2]s counts WHERE counts.item_id = %[1]s.item_id), %[1]s.labels)
		ELSE %[1]s.labels END AS labels`, base, counts)
	return "item_id, is_hidden, categories, time_stamp, " + labels + ", comment"
}

func splitTwitchViewCount(item Item) (Item, *int64, error) {
	object, ok := item.Labels.(map[string]any)
	if !ok || object["platform"] != "twitch" || object["externalKind"] != "Live" {
		return item, nil, nil
	}
	// Parse just the scalar with UseNumber so int64 precision is retained.
	// Other providers and large embedding arrays need no extra JSON round trip.
	encoded, err := json.Marshal(object["viewCount"])
	if err != nil {
		return Item{}, nil, err
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	if err = decoder.Decode(&value); err != nil {
		return Item{}, nil, err
	}
	number, ok := value.(json.Number)
	if !ok {
		return item, nil, nil
	}
	count, err := number.Int64()
	if err != nil || count < 0 {
		return item, nil, nil
	}
	// Only a top-level key is removed; shared vectors are read, never changed.
	cold := make(map[string]any, len(object)-1)
	for key, value := range object {
		if key != "viewCount" {
			cold[key] = value
		}
	}
	item.Labels = cold
	return item, &count, nil
}

func (d *SQLDatabase) saveViewCount(tx *gorm.DB, itemId string, count *int64, projectionHash []byte, replaceHash bool) error {
	if count == nil {
		return tx.Table(d.itemViewCountsTable()).Where("item_id = ?", itemId).Delete(&sqlItemViewCount{}).Error
	}
	columns := []string{"view_count"}
	if replaceHash {
		columns = append(columns, "projection_hash")
	}
	return tx.Table(d.itemViewCountsTable()).Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "item_id"}},
		DoUpdates: clause.AssignmentColumns(columns),
	}).Create(&sqlItemViewCount{ItemId: itemId, ViewCount: *count, ProjectionHash: projectionHash}).Error
}

func sameSQLItemProjection(left, right SQLItem) bool {
	return left.ItemId == right.ItemId &&
		left.IsHidden == right.IsHidden &&
		sameJSONProjection(left.Categories, right.Categories) &&
		left.Timestamp.Equal(right.Timestamp) &&
		sameJSONProjection(left.Labels, right.Labels) &&
		left.Comment == right.Comment
}

// MySQL's JSON columns may return semantically identical objects with a
// different key order or insignificant whitespace than encoding/json emitted.
// Preserve number spellings while canonicalizing so large integer labels are
// never made equal through float64 rounding.
func sameJSONProjection(left, right string) bool {
	if left == right {
		return true
	}
	canonical := func(value string) (string, bool) {
		var decoded any
		decoder := json.NewDecoder(strings.NewReader(value))
		decoder.UseNumber()
		if err := decoder.Decode(&decoded); err != nil {
			return "", false
		}
		encoded, err := json.Marshal(decoded)
		return string(encoded), err == nil
	}
	canonicalLeft, leftOK := canonical(left)
	canonicalRight, rightOK := canonical(right)
	return leftOK && rightOK && canonicalLeft == canonicalRight
}

func (d *SQLDatabase) insertItemsWithViewCounts(ctx context.Context, items []Item) error {
	// Preserve first-duplicate-wins semantics and lock in a consistent order.
	unique := make(map[string]Item, len(items))
	for _, item := range items {
		if _, exists := unique[item.ItemId]; !exists {
			unique[item.ItemId] = item
		}
	}
	ids := make([]string, 0, len(unique))
	for id := range unique {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	// Keep ordinary provider batches bulk writes rather than turning every
	// YouTube ingestion batch into per-item read/write transactions.
	containsTwitchLive := false
	for _, item := range unique {
		labels, ok := item.Labels.(map[string]any)
		if ok && labels["platform"] == "twitch" && labels["externalKind"] == "Live" {
			containsTwitchLive = true
			break
		}
	}
	decisions := viewCountDecisions{}
	if !containsTwitchLive {
		rows := make(map[string]SQLItem, len(ids))
		for _, id := range ids {
			row := NewSQLItem(unique[id])
			row.Timestamp = d.normalizeTimestampForWrite(row.Timestamp)
			rows[id] = row
		}
		decisions.evaluate(ViewCountBranchUnsplittable, len(ids))
		projectionDecisions := itemProjectionDecisions{}
		err := d.gormDB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
			var existing []SQLItem
			if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
				Where("item_id IN ?", ids).Order("item_id").Find(&existing).Error; err != nil {
				return err
			}
			byID := make(map[string]SQLItem, len(existing))
			for _, row := range existing {
				byID[row.ItemId] = row
			}
			changed := make([]SQLItem, 0, len(rows))
			for _, id := range ids {
				current, exists := byID[id]
				switch {
				case !exists:
					projectionDecisions.evaluate(ItemProjectionBranchCreated)
					changed = append(changed, rows[id])
				case sameSQLItemProjection(current, rows[id]):
					projectionDecisions.evaluate(ItemProjectionBranchUnchanged)
				default:
					projectionDecisions.evaluate(ItemProjectionBranchChanged)
					changed = append(changed, rows[id])
				}
			}
			if len(changed) > 0 {
				if err := tx.Clauses(clause.OnConflict{
					Columns:   []clause.Column{{Name: "item_id"}},
					DoUpdates: clause.AssignmentColumns([]string{"is_hidden", "categories", "time_stamp", "labels", "comment"}),
				}).Create(changed).Error; err != nil {
					return err
				}
			}
			return tx.Table(d.itemViewCountsTable()).Where("item_id IN ?", ids).Delete(&sqlItemViewCount{}).Error
		})
		if err != nil {
			return err
		}
		decisions.committed()
		projectionDecisions.committed()
		return nil
	}
	err := d.gormDB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		for _, id := range ids {
			cold, count, err := splitTwitchViewCount(unique[id])
			if err != nil {
				return err
			}
			row := NewSQLItem(cold)
			row.Timestamp = d.normalizeTimestampForWrite(row.Timestamp)

			encoded, err := json.Marshal(row)
			if err != nil {
				return err
			}
			fingerprint := sha256.Sum256(encoded)
			state := viewCountWriteState{marked: IsTwitchLiveProjection(ctx), splittable: count != nil}
			var current SQLItem
			err = tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("item_id = ?", id).Take(&current).Error
			state.exists = err == nil
			if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
				return err
			}
			if state.exists && count != nil {
				var previous sqlItemViewCount
				err = tx.Table(d.itemViewCountsTable()).Where("item_id = ?", id).Take(&previous).Error
				if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
					return err
				}
				state.countRow = err == nil
				state.sameFingerprint = state.countRow && bytes.Equal(previous.ProjectionHash, fingerprint[:])
				var currentLabels map[string]any
				if json.Unmarshal([]byte(current.Labels), &currentLabels) == nil {
					_, state.legacyLabel = currentLabels["viewCount"]
				}
			}
			// The branch is the decision: only the short path skips the item row.
			branch := classifyViewCountWrite(state)
			decisions.evaluate(branch, 1)
			if branch != ViewCountBranchShortPath {
				if err = tx.Clauses(clause.OnConflict{
					Columns:   []clause.Column{{Name: "item_id"}},
					DoUpdates: clause.AssignmentColumns([]string{"is_hidden", "categories", "time_stamp", "labels", "comment"}),
				}).Create(&row).Error; err != nil {
					return err
				}
			}
			if err = d.saveViewCount(tx, id, count, fingerprint[:], true); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	decisions.committed()
	return nil
}

func (d *SQLDatabase) modifyItemWithViewCount(ctx context.Context, itemId string, patch ItemPatch) error {
	return d.gormDB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var row SQLItem
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("item_id = ?", itemId).Take(&row).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return nil
			} // retain ModifyItem's absent-item semantics
			return err
		}
		cold, count, err := splitTwitchViewCount(Item{ItemId: itemId, Labels: patch.Labels})
		if err != nil {
			return err
		}
		patch.Labels = cold.Labels
		if err = d.modifyItemColumns(tx, itemId, patch); err != nil {
			return err
		}
		return d.saveViewCount(tx, itemId, count, nil, false)
	})
}

// AtomicItemLabelPatcher is optional. The validator runs against the actual
// merged object while the item row is locked, retaining endpoint size/vector
// checks without a read/modify/write race with a viewer-count projection.
type AtomicItemLabelPatcher interface {
	PatchItemLabelsAtomic(context.Context, string, map[string]any, func(Item) error) (Item, bool, error)
}

func (d *SQLDatabase) PatchItemLabelsAtomic(ctx context.Context, itemId string, patch map[string]any, validate func(Item) error) (Item, bool, error) {
	if !d.separateTwitchViewCounts {
		return Item{}, false, nil
	}
	var item Item
	changed := false
	err := d.gormDB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Lock the base row first, matching all full-item/count writers.
		var row SQLItem
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("item_id = ?", itemId).Take(&row).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return storage.ErrNotFound
			}
			return err
		}
		if err := tx.Table(d.ItemsTable()).Select(d.itemSelectColumns()).Where("item_id = ?", itemId).Take(&item).Error; err != nil {
			return err
		}
		// Seed legacy rows with their pre-patch source fingerprint. Subsequent
		// count-only projections must not overwrite this patch's embedding.
		originalCold, _, err := splitTwitchViewCount(item)
		if err != nil {
			return err
		}
		originalRow := NewSQLItem(originalCold)
		originalRow.Timestamp = d.normalizeTimestampForWrite(originalRow.Timestamp)
		originalBytes, err := json.Marshal(originalRow)
		if err != nil {
			return err
		}
		originalHash := sha256.Sum256(originalBytes)
		labels, ok := item.Labels.(map[string]any)
		if item.Labels == nil {
			labels = make(map[string]any)
			ok = true
		}
		if !ok {
			return errors.New("item labels are not a JSON object and cannot be merged")
		}
		for key, value := range patch {
			if value == nil {
				delete(labels, key)
			} else {
				labels[key] = value
			}
		}
		item.Labels = labels
		if err := validate(item); err != nil {
			return err
		}
		cold, count, err := splitTwitchViewCount(item)
		if err != nil {
			return err
		}
		encoded, err := json.Marshal(cold.Labels)
		if err != nil {
			return err
		}
		if sameJSONProjection(row.Labels, string(encoded)) {
			// The merged labels may have changed only the separately stored
			// view count (including deleting it), so preserve that transition
			// while avoiding an identical base-row update.
			return d.saveViewCount(tx, itemId, count, originalHash[:], false)
		}
		changed = true
		if err = tx.Model(&SQLItem{}).Where("item_id = ?", itemId).Update("labels", string(encoded)).Error; err != nil {
			return err
		}
		return d.saveViewCount(tx, itemId, count, originalHash[:], false)
	})
	if err == nil {
		result := "unchanged"
		if changed {
			result = "changed"
		}
		ItemLabelPatchCommitsTotal.WithLabelValues(result).Inc()
	}
	return item, true, errors.WithStack(err)
}
