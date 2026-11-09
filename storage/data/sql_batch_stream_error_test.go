package data

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorse-io/gorse/storage"
	mysqlgorm "gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var errLateItemStream = errors.New("injected late item stream error")

// itemStreamFixture yields one row and then ends the stream the way mode says.
type itemStreamFixture struct {
	mode                      string
	idOnly                    bool
	queries, closed, returned atomic.Int32
	secondNext                chan struct{}
}

type itemStreamConnector struct{ fixture *itemStreamFixture }
type itemStreamDriver struct{}
type itemStreamConn struct{ fixture *itemStreamFixture }
type itemStreamRows struct {
	fixture *itemStreamFixture
	ctx     context.Context
	calls   int
}

func (c itemStreamConnector) Connect(context.Context) (driver.Conn, error) {
	return &itemStreamConn{c.fixture}, nil
}
func (itemStreamConnector) Driver() driver.Driver { return itemStreamDriver{} }
func (itemStreamDriver) Open(string) (driver.Conn, error) {
	return nil, errors.New("use the connector")
}
func (*itemStreamConn) Prepare(string) (driver.Stmt, error) {
	return nil, errors.New("unexpected prepare")
}
func (*itemStreamConn) Close() error              { return nil }
func (*itemStreamConn) Begin() (driver.Tx, error) { return nil, errors.New("unexpected transaction") }
func (c *itemStreamConn) QueryContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	if !strings.HasPrefix(query, "SELECT ") || len(args) != 2 {
		return nil, errors.New("unexpected query")
	}
	c.fixture.queries.Add(1)
	return &itemStreamRows{fixture: c.fixture, ctx: ctx}, nil
}

func (r *itemStreamRows) Columns() []string {
	if r.fixture.idOnly {
		return []string{"item_id"}
	}
	return []string{"item_id", "is_hidden", "categories", "time_stamp", "labels", "comment"}
}

func (r *itemStreamRows) Close() error {
	r.fixture.closed.Add(1)
	return nil
}

func (r *itemStreamRows) Next(values []driver.Value) error {
	r.calls++
	if r.calls == 1 {
		r.fixture.returned.Add(1)
		if r.fixture.idOnly {
			values[0] = "item1"
			return nil
		}
		copy(values, []driver.Value{
			"item1", int64(0), []byte(`["a"]`), time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
			[]byte(`{"embedding":[0.25,0.5],"revision":1}`), "comment",
		})
		return nil
	}
	if r.calls == 2 {
		close(r.fixture.secondNext)
	}
	switch r.fixture.mode {
	case "late":
		return errLateItemStream
	case "cancel", "deadline":
		<-r.ctx.Done()
		return r.ctx.Err()
	default:
		return io.EOF
	}
}

func TestBatchGetItemsReportsStreamErrorsInsteadOfPartialResults(t *testing.T) {
	type result struct {
		items []Item
		err   error
	}
	for _, idOnly := range []bool{false, true} {
		for _, mode := range []string{"ok", "late", "cancel", "deadline"} {
			name := "full/" + mode
			if idOnly {
				name = "ids/" + mode
			}
			t.Run(name, func(t *testing.T) {
				fixture := &itemStreamFixture{mode: mode, idOnly: idOnly, secondNext: make(chan struct{})}
				db := sql.OpenDB(itemStreamConnector{fixture})
				defer db.Close()
				gormDB, err := gorm.Open(
					mysqlgorm.New(mysqlgorm.Config{Conn: db, SkipInitializeWithVersion: true}),
					&gorm.Config{DisableAutomaticPing: true, Logger: logger.Default.LogMode(logger.Silent)})
				if err != nil {
					t.Fatal(err)
				}
				database := &SQLDatabase{client: db, gormDB: gormDB, driver: MySQL, TablePrefix: storage.TablePrefix("fixture_")}

				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				if mode == "deadline" {
					ctx, cancel = context.WithTimeout(t.Context(), 500*time.Millisecond)
					defer cancel()
				}

				done := make(chan result, 1)
				go func() {
					items, err := database.BatchGetItems(ctx, []string{"item1", "missing"}, GetOptions{ReturnId: idOnly})
					done <- result{items, err}
				}()
				if mode == "cancel" || mode == "deadline" {
					select {
					case <-fixture.secondNext:
						if mode == "cancel" {
							cancel()
						}
					case <-time.After(time.Second):
						t.Fatal("reader never reached the end of the stream")
					}
				}

				var got result
				select {
				case got = <-done:
				case <-time.After(2 * time.Second):
					t.Fatal("reader did not return")
				}

				if mode == "ok" {
					if got.err != nil || len(got.items) != 1 || got.items[0].ItemId != "item1" {
						t.Fatalf("clean end of stream changed: %#v, %v", got.items, got.err)
					}
					if !idOnly {
						labels, err := json.Marshal(got.items[0].Labels)
						if err != nil || string(labels) != `{"embedding":[0.25,0.5],"revision":1}` {
							t.Fatalf("labels changed: %s, %v", labels, err)
						}
					}
				} else {
					want := errLateItemStream
					if mode == "cancel" {
						want = context.Canceled
					}
					if mode == "deadline" {
						want = context.DeadlineExceeded
					}
					if !errors.Is(got.err, want) {
						t.Fatalf("stream error lost: want %v, got %v (items=%d)", want, got.err, len(got.items))
					}
					if got.items != nil {
						t.Fatalf("returned %d partial items alongside a stream error", len(got.items))
					}
				}
				if fixture.queries.Load() != 1 || fixture.returned.Load() != 1 || fixture.closed.Load() != 1 {
					t.Fatalf("unexpected lifecycle: queries=%d returned=%d closed=%d",
						fixture.queries.Load(), fixture.returned.Load(), fixture.closed.Load())
				}
			})
		}
	}
}
