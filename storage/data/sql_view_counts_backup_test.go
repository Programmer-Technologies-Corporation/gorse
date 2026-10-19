package data_test

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	mysqldriver "github.com/go-sql-driver/mysql"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/model/ctr"
	"github.com/gorse-io/gorse/storage"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"
)

// This file is the only place that runs real mysqldump/mysql client tools. It
// is opt-in through TWITCH_COUNT_BACKUP_MYSQL_URI and fails closed: the server
// must be loopback, the dump covers only this test's prefixed synthetic tables,
// and the restore target is a uniquely named database the test creates and
// drops itself. Credentials travel through MYSQL_PWD and are never printed;
// dump contents are never logged or included in assertion messages.

const (
	backupContainerPrefix = "vh-count-tests-"
	backupDatabasePrefix  = "vc_backup_"
)

type backupTools struct {
	cfg           *mysqldriver.Config
	container     string // optional container running the client tools (docker exec)
	dockerContext string // optional docker context for that container
}

// parseBackupTarget rejects anything that is not an explicitly local,
// synthetic MySQL target before a single client tool is started.
func parseBackupTarget(uri, container, dockerContext string) (backupTools, error) {
	if !strings.HasPrefix(uri, storage.MySQLPrefix) {
		return backupTools{}, errors.New("backup target must be a mysql:// URI")
	}
	cfg, err := mysqldriver.ParseDSN(uri[len(storage.MySQLPrefix):])
	if err != nil {
		return backupTools{}, errors.WithStack(err)
	}
	if cfg.Net != "tcp" {
		return backupTools{}, fmt.Errorf("backup target must use tcp, got %q", cfg.Net)
	}
	host := cfg.Addr
	if h, _, err := net.SplitHostPort(cfg.Addr); err == nil {
		host = h
	}
	switch host {
	case "127.0.0.1", "localhost", "::1":
	default:
		return backupTools{}, fmt.Errorf("backup target host %q is not loopback", host)
	}
	if cfg.DBName != "gorse_counts_test" {
		return backupTools{}, errors.New("backup source must be the dedicated gorse_counts_test database")
	}
	if container != "" && !strings.HasPrefix(container, backupContainerPrefix) {
		return backupTools{}, fmt.Errorf("backup tool container must be a disposable %q container", backupContainerPrefix)
	}
	if dockerContext != "" && container == "" {
		return backupTools{}, errors.New("a docker context is only meaningful with a backup tool container")
	}
	return backupTools{cfg: cfg, container: container, dockerContext: dockerContext}, nil
}

// run executes one client tool with stdin and returns stdout. Host tools
// connect over loopback TCP; container tools use the container's own socket.
func (b backupTools) run(t *testing.T, stdin []byte, tool string, args ...string) []byte {
	t.Helper()
	var cmd *exec.Cmd
	if b.container != "" {
		docker := []string{}
		if b.dockerContext != "" {
			docker = append(docker, "--context", b.dockerContext)
		}
		docker = append(docker, "exec", "-i", "-e", "MYSQL_PWD", b.container, tool, "--user="+b.cfg.User)
		cmd = exec.Command("docker", append(docker, args...)...)
	} else {
		path, err := exec.LookPath(tool)
		require.NoError(t, err, "%s must be installed for the opt-in backup regression", tool)
		host, port, err := net.SplitHostPort(b.cfg.Addr)
		if err != nil {
			host, port = b.cfg.Addr, "3306"
		}
		cmd = exec.Command(path, append([]string{"--protocol=TCP", "--host=" + host, "--port=" + port, "--user=" + b.cfg.User}, args...)...)
	}
	cmd.Env = append(os.Environ(), "MYSQL_PWD="+b.cfg.Passwd)
	cmd.Stdin = bytes.NewReader(stdin)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	diagnostics := stderr.String()
	if b.cfg.Passwd != "" {
		diagnostics = strings.ReplaceAll(diagnostics, b.cfg.Passwd, "***")
	}
	require.NoError(t, err, "%s failed: %s", tool, diagnostics)
	return stdout.Bytes()
}

func TestBackupTargetFailsClosed(t *testing.T) {
	for name, uri := range map[string]string{
		"remote host":   "mysql://root:pw@tcp(db.internal:3306)/gorse",
		"public ip":     "mysql://root:pw@tcp(203.0.113.9:3306)/gorse",
		"unix socket":   "mysql://root:pw@unix(/var/run/mysqld/mysqld.sock)/gorse",
		"no database":   "mysql://root:pw@tcp(127.0.0.1:3306)/",
		"live database": "mysql://root:pw@tcp(127.0.0.1:3306)/gorse",
		"not mysql":     "postgres://gorse:pw@127.0.0.1/gorse",
		"injected name": "mysql://root:pw@tcp(127.0.0.1:3306)/gorse`;DROP",
	} {
		t.Run(name, func(t *testing.T) {
			_, err := parseBackupTarget(uri, "", "")
			require.Error(t, err)
		})
	}
	local := "mysql://root:pw@tcp(127.0.0.1:3306)/gorse_counts_test"
	_, err := parseBackupTarget(local, "mysql-prod", "")
	require.Error(t, err, "only disposable test containers may run the tools")
	_, err = parseBackupTarget(local, "", "desktop-linux")
	require.Error(t, err, "a context without a container is a misconfiguration")
	tools, err := parseBackupTarget(local, backupContainerPrefix+"local", "desktop-linux")
	require.NoError(t, err)
	require.Equal(t, "gorse_counts_test", tools.cfg.DBName)
	_, err = parseBackupTarget("mysql://root:pw@tcp(localhost)/gorse_counts_test", "", "")
	require.NoError(t, err)
}

func TestMySQLTwitchViewCountsBackupRestore(t *testing.T) {
	uri := os.Getenv("TWITCH_COUNT_BACKUP_MYSQL_URI")
	if uri == "" {
		t.Skip("TWITCH_COUNT_BACKUP_MYSQL_URI is not set; the mysqldump/restore regression is opt-in")
	}
	tools, err := parseBackupTarget(uri, os.Getenv("TWITCH_COUNT_BACKUP_TOOL_CONTAINER"), os.Getenv("TWITCH_COUNT_BACKUP_DOCKER_CONTEXT"))
	require.NoError(t, err)
	ctx := context.Background()
	marked := data.WithTwitchLiveProjection(ctx)
	admin, err := sql.Open("mysql", tools.cfg.FormatDSN())
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, admin.Close()) })
	var serverUUID string
	require.NoError(t, admin.QueryRow("SELECT @@server_uuid").Scan(&serverUUID))
	toolUUID := tools.run(t, nil, "mysql", "--batch", "--skip-column-names", "--execute=SELECT @@server_uuid", tools.cfg.DBName)
	require.True(t, strings.TrimSpace(string(toolUUID)) == serverUUID, "backup tools and Go must address the same isolated MySQL server")
	prefix := fmt.Sprintf("vcb_%d_", time.Now().UnixNano())
	legacy, err := data.Open(uri, prefix, storage.WithIsolationLevel("READ-COMMITTED"))
	require.NoError(t, err)
	require.NoError(t, legacy.Init())
	source, err := data.Open(uri, prefix, storage.WithIsolationLevel("READ-COMMITTED"), storage.WithSeparateTwitchViewCounts(true))
	require.NoError(t, err)
	require.NoError(t, source.Init())
	require.NoError(t, source.Reconcile(config.SearchConfig{Columns: []string{"item.Comment"}}))
	t.Cleanup(func() {
		require.NoError(t, source.Purge())
		require.NoError(t, source.Close())
		require.NoError(t, legacy.Close())
	})

	// Mixed representations: a converted live item, an untouched legacy live
	// item, an unsplittable VOD, plus a user and feedback for the other tables.
	timestamp := time.Date(2026, 10, 4, 12, 0, 0, 0, time.UTC)
	require.NoError(t, legacy.BatchInsertItems(ctx, []data.Item{liveItem("legacy", 42)}))
	require.NoError(t, source.BatchInsertItems(marked, []data.Item{liveItem("live", 8731)}))
	vod := liveItem("vod", 7)
	vod.Labels.(map[string]any)["externalKind"] = "Video"
	require.NoError(t, source.BatchInsertItems(ctx, []data.Item{vod}))
	require.NoError(t, source.BatchInsertUsers(ctx, []data.User{{UserId: "u", Labels: map[string]any{"tier": "synthetic"}, Comment: "synthetic user"}}))
	require.NoError(t, source.BatchInsertFeedback(ctx, []data.Feedback{{FeedbackKey: data.FeedbackKey{FeedbackType: "watch", UserId: "u", ItemId: "live"}, Timestamp: timestamp}}, false, false, true))
	expected, err := source.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(8731), viewCount(t, expected))

	// Dump exactly the four data tables of this prefix. The contents stay in
	// memory and a temp file; assertions only report booleans.
	tables := []string{prefix + "items", prefix + "users", prefix + "feedback", prefix + "item_view_counts"}
	dump := tools.run(t, nil, "mysqldump", append([]string{"--single-transaction", "--no-tablespaces", "--skip-triggers", "--set-gtid-purged=OFF", tools.cfg.DBName}, tables...)...)
	for _, table := range tables {
		require.True(t, bytes.Contains(dump, []byte("CREATE TABLE `"+table+"`")), "dump lacks table %s", table)
	}
	require.False(t, bytes.Contains(dump, []byte("CREATE DATABASE")), "a table-level dump must not carry database statements")
	dumpPath := t.TempDir() + "/backup.sql"
	require.NoError(t, os.WriteFile(dumpPath, dump, 0600))

	// Restore into a uniquely named synthetic database; only that database is
	// ever dropped, and only by this test.
	synthetic := fmt.Sprintf("%s%d", backupDatabasePrefix, time.Now().UnixNano())
	_, err = admin.Exec("CREATE DATABASE `" + synthetic + "` CHARACTER SET utf8mb4")
	require.NoError(t, err)
	t.Cleanup(func() {
		require.True(t, strings.HasPrefix(synthetic, backupDatabasePrefix))
		_, err := admin.Exec("DROP DATABASE IF EXISTS `" + synthetic + "`")
		require.NoError(t, err)
	})
	fromFile, err := os.ReadFile(dumpPath)
	require.NoError(t, err)
	tools.run(t, fromFile, "mysql", synthetic)

	restoredCfg := *tools.cfg
	restoredCfg.DBName = synthetic
	restoredURI := storage.MySQLPrefix + restoredCfg.FormatDSN()
	restored, err := data.Open(restoredURI, prefix, storage.WithIsolationLevel("READ-COMMITTED"), storage.WithSeparateTwitchViewCounts(true))
	require.NoError(t, err)
	require.NoError(t, restored.Init(), "Init on a restored schema is idempotent")
	require.NoError(t, restored.Reconcile(config.SearchConfig{Columns: []string{"item.Comment"}}))
	oldReader, err := data.Open(restoredURI, prefix, storage.WithIsolationLevel("READ-COMMITTED"))
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, restored.Close())
		require.NoError(t, oldReader.Close())
	})
	countRows := func() int {
		var rows int
		require.NoError(t, admin.QueryRow("SELECT COUNT(*) FROM `"+synthetic+"`.`"+prefix+"item_view_counts`").Scan(&rows))
		return rows
	}
	var constraints int
	require.NoError(t, admin.QueryRow("SELECT COUNT(*) FROM information_schema.REFERENTIAL_CONSTRAINTS WHERE CONSTRAINT_SCHEMA=? AND TABLE_NAME=?", synthetic, prefix+"item_view_counts").Scan(&constraints))
	require.Equal(t, 1, constraints, "the cascade foreign key must survive dump and restore")
	require.Equal(t, 1, countRows(), "only the converted live item owns a count row")

	// Every reader, the FM feature conversion and the embedding conversion see
	// the pre-backup representation on the restored copy.
	got, err := restored.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, expected, got)
	require.Equal(t, sortedFeatures(expected.Labels), sortedFeatures(got.Labels))
	require.Equal(t, ctr.ConvertEmbeddings(expected.Labels), ctr.ConvertEmbeddings(got.Labels))
	byId := func(items []data.Item) map[string]float64 {
		out := map[string]float64{}
		for _, item := range items {
			out[item.ItemId] = viewCount(t, item)
		}
		return out
	}
	want := map[string]float64{"live": 8731, "legacy": 42, "vod": 7}
	batch, err := restored.BatchGetItems(ctx, []string{"live", "legacy", "vod"}, data.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, want, byId(batch))
	_, paged, err := restored.GetItems(ctx, "", 10, nil)
	require.NoError(t, err)
	require.Equal(t, want, byId(paged))
	latest, err := restored.GetLatestItems(ctx, 10, nil, nil)
	require.NoError(t, err)
	require.Equal(t, want, byId(latest))
	stream, errs := restored.GetItemStream(ctx, 10, nil)
	var streamed []data.Item
	for chunk := range stream {
		streamed = append(streamed, chunk...)
	}
	require.NoError(t, <-errs)
	require.Equal(t, want, byId(streamed))
	search, err := restored.SearchItems(ctx, "synthetic", 10)
	require.NoError(t, err)
	var searched []data.Item
	for _, hit := range search {
		searched = append(searched, hit.Item)
	}
	require.Equal(t, want, byId(searched))
	user, err := restored.GetUser(ctx, "u")
	require.NoError(t, err)
	require.Equal(t, "synthetic user", user.Comment)
	feedback, err := restored.GetUserFeedback(ctx, "u", nil)
	require.NoError(t, err)
	require.Len(t, feedback, 1)
	require.Equal(t, "live", feedback[0].ItemId)

	// The restored projection hash still makes an unchanged source eligible.
	expectBranch(t, data.ViewCountBranchShortPath, func() {
		require.NoError(t, restored.BatchInsertItems(marked, []data.Item{liveItem("live", 8732)}))
	})

	// Rollback rehearsal with writers quiesced: copy hot counts back into
	// labels, let a split-OFF reader verify them, and keep the count table.
	split, err := restored.GetItem(ctx, "live")
	require.NoError(t, err)
	_, err = admin.Exec("UPDATE `" + synthetic + "`.`" + prefix + "items` i JOIN `" + synthetic + "`.`" + prefix + "item_view_counts` c ON i.item_id=c.item_id SET i.labels=JSON_SET(i.labels,'$.viewCount',c.view_count)")
	require.NoError(t, err)
	rolledBack, err := oldReader.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(8732), viewCount(t, rolledBack))
	require.Equal(t, split, rolledBack, "old and split readers agree after copy-back")
	require.Equal(t, sortedFeatures(split.Labels), sortedFeatures(rolledBack.Labels))
	require.Equal(t, 1, countRows(), "rollback preserves the count table")

	// Re-enable: the legacy label forces normalization despite the old hash.
	expectBranch(t, data.ViewCountBranchLegacyConversion, func() {
		require.NoError(t, restored.BatchInsertItems(marked, []data.Item{liveItem("live", 8733)}))
	})
	current, err := restored.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(8733), viewCount(t, current))
	var present int
	require.NoError(t, admin.QueryRow("SELECT JSON_CONTAINS_PATH(labels,'one','$.viewCount') FROM `"+synthetic+"`.`"+prefix+"items` WHERE item_id='live'").Scan(&present))
	require.Zero(t, present)
	require.NoError(t, restored.DeleteItem(ctx, "live"))
	require.Zero(t, countRows(), "the restored cascade removes the count row")
}
