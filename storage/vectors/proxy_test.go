// Copyright 2026 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vectors

import (
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/gorse-io/gorse/common/log"
	"github.com/gorse-io/gorse/storage"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
)

type ProxyTestSuite struct {
	vectorsTestSuite
	backend    Database
	server     *ProxyServer
	clientConn *grpc.ClientConn
}

func (suite *ProxyTestSuite) SetupSuite() {
	log.SetTestLogger(suite.T())
	var err error
	path := fmt.Sprintf("xvec://%s/vectors", suite.T().TempDir())
	suite.backend, err = Open(path, "gorse_")
	suite.NoError(err)
	suite.NoError(suite.backend.Init())
	lis, err := net.Listen("tcp", "localhost:0")
	suite.NoError(err)
	suite.server = NewProxyServer(suite.backend)
	go func() {
		err = suite.server.Serve(lis)
		suite.NoError(err)
	}()
	suite.clientConn, err = grpc.Dial(lis.Addr().String(), grpc.WithInsecure())
	suite.NoError(err)
	suite.Database = NewProxyClient(suite.clientConn)
}

func (suite *ProxyTestSuite) TearDownSuite() {
	suite.server.Stop()
	suite.NoError(suite.clientConn.Close())
	suite.NoError(suite.backend.Close())
}

func TestProxy(t *testing.T) {
	suite.Run(t, new(ProxyTestSuite))
}

// VideoHub fork: sentinel errors survive the gRPC hop for every method, not
// only DescribeCollection.
func (suite *ProxyTestSuite) TestSentinelErrors() {
	ctx := suite.T().Context()
	_, err := suite.Database.GetVectors(ctx, "missing", []string{"a"})
	suite.ErrorIs(err, storage.ErrNotFound)
	_, err = suite.Database.QueryVectors(ctx, "missing", Vector{Values: make([]float32, defaultVectorSize)}, nil, 1)
	suite.ErrorIs(err, storage.ErrNotFound)
	_, err = suite.Database.CountVectors(ctx, "missing")
	suite.ErrorIs(err, storage.ErrNotFound)
	suite.ErrorIs(suite.Database.DeleteVectors(ctx, "missing", time.Now()), storage.ErrNotFound)
	suite.ErrorIs(suite.Database.DeleteCollection(ctx, "missing"), storage.ErrNotFound)
	suite.NoError(suite.Database.AddCollection(ctx, "dup", defaultVectorSize, Cosine, VectorConfig{}))
	suite.ErrorIs(suite.Database.AddCollection(ctx, "dup", defaultVectorSize, Cosine, VectorConfig{}), storage.ErrAlreadyExists)
}
