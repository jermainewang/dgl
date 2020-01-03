#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <dmlc/omp.h>

#include "../c_api_common.h"
#include "../array/common.h"  // for ATEN_FLOAT_TYPE_SWITCH

using namespace dgl::runtime;

namespace dgl {

GraphPtr TestSampleImpl(ImmutableGraphPtr g, IdArray seed_nodes, int64_t fanout, std::string edge_dir) {
  const dgl_id_t* nids = static_cast<const dgl_id_t*>(seed_nodes->data);
  const int64_t L = seed_nodes->shape[0];

  aten::CSRMatrix csr = (edge_dir == "in")? g->GetInCSR()->ToCSRMatrix() : g->GetOutCSR()->ToCSRMatrix();
  const int64_t* indptr_data = static_cast<const int64_t*>(csr.indptr->data);
  const int64_t* indices_data = static_cast<const int64_t*>(csr.indices->data);

  IdArray rstsrc = aten::NewIdArray(fanout * L);
  IdArray rstdst = aten::NewIdArray(fanout * L);
  dgl_id_t* rstsrcdata = static_cast<dgl_id_t*>(rstsrc->data);
  dgl_id_t* rstdstdata = static_cast<dgl_id_t*>(rstdst->data);

#pragma omp parallel for
  for (int64_t i = 0; i < L; ++i) {
    const dgl_id_t nid = nids[i];
    const int64_t off = indptr_data[nid];
    const int64_t len = indptr_data[nid + 1] - off;
    for (int64_t j = 0; j < fanout; ++j) {
      rstsrcdata[i * fanout + j] = indices_data[off + RandomEngine::ThreadLocal()->RandInt(len)];
      rstdstdata[i * fanout + j] = nid;
    }
  }

  return ImmutableGraph::CreateFromCOO(g->NumVertices(), rstsrc, rstdst);
}

DGL_REGISTER_GLOBAL("sampling._CAPI_NeighborSamplingNew")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    // arguments
    const GraphRef g = args[0];
    const IdArray seed_nodes = args[1];
    const int64_t fanout = args[2];
    const std::string edge_dir = args[3];
    const NDArray probability = args[4];
    const bool replace = args[5];

    auto gptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
    CHECK(gptr) << "sampling isn't implemented in mutable graph";

    CHECK(aten::IsValidIdArray(seed_nodes));
    CHECK_EQ(seed_nodes->ctx.device_type, kDLCPU)
      << "NeighborSampler only support CPU sampling";

    CHECK(probability->dtype.code == kDLFloat)
      << "transition probability must be float";
    CHECK(probability->ndim == 1)
      << "transition probability must be a 1-dimensional vector";
    CHECK_EQ(probability->ctx.device_type, kDLCPU)
      << "NeighborSampling only support CPU sampling";

    GraphPtr ret;

    ATEN_FLOAT_TYPE_SWITCH(
      probability->dtype,
      FloatType,
      "transition probability",
      {
        const FloatType *prob;

        if (probability->ndim == 1 && probability->shape[0] == 0) {
          prob = nullptr;
        } else {
          CHECK(probability->shape[0] == gptr->NumEdges())
            << "transition probability must have same number of elements as edges";
          CHECK(probability.IsContiguous())
            << "transition probability must be contiguous tensor";
          prob = static_cast<const FloatType *>(probability->data);
        }

        ret = TestSampleImpl(gptr, seed_nodes, fanout, edge_dir);

    });

    *rv = GraphRef(ret);
  });

List<GraphRef> CompactGraphs(List<GraphRef> graphs) {
  // estimate the number of buckets in the hash
  int64_t est_size = 0;
  for (size_t i = 0; i < graphs.size(); ++i)
    est_size += graphs[i]->NumEdges();
  std::unordered_map<dgl_id_t, dgl_id_t> hash(est_size);
  std::vector<IdArray> newsrc(graphs.size()), newdst(graphs.size());
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto& earr = graphs[i]->Edges();
    const int64_t len = earr.src->shape[0];
    newsrc[i] = aten::NewIdArray(len);
    newdst[i] = aten::NewIdArray(len);
    const dgl_id_t* srcdata = static_cast<dgl_id_t*>(earr.src->data);
    const dgl_id_t* dstdata = static_cast<dgl_id_t*>(earr.dst->data);
    dgl_id_t* newsrcdata = static_cast<dgl_id_t*>(newsrc[i]->data);
    dgl_id_t* newdstdata = static_cast<dgl_id_t*>(newdst[i]->data);
    for (auto j = 0; j < len; ++j) {
      auto it = hash.find(srcdata[j]);
      if (it == hash.end()) {
        const int64_t newid = hash.size();
        hash[srcdata[j]] = newid;
        newsrcdata[j] = newid;
      } else {
        newsrcdata[j] = it->second;
      }
      it = hash.find(dstdata[j]);
      if (it == hash.end()) {
        const int64_t newid = hash.size();
        hash[dstdata[j]] = newid;
        newdstdata[j] = newid;
      } else {
        newdstdata[j] = it->second;
      }
    }
  }
  const int64_t num_nodes = hash.size();
  std::vector<GraphRef> ret(graphs.size());
  for (size_t i = 0; i < graphs.size(); ++i) {
    ret[i] = GraphRef(ImmutableGraph::CreateFromCOO(num_nodes, newsrc[i], newdst[i]));
  }
  return List<GraphRef>(ret);
}

DGL_REGISTER_GLOBAL("sampling._CAPI_CompactGraphs")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    // arguments
    const List<GraphRef> graphs = args[0];
    *rv = CompactGraphs(graphs);
  });

}  // namespace dgl
