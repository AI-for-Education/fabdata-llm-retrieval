from collections import defaultdict
from types import SimpleNamespace
import time

import numpy as np
from fdllm import get_caller
from fdllm.chat import ChatController
from fdllm.decorators import delayedretry
from redis.exceptions import ConnectionError

from ..models.models import Query, DocumentMetadataFilter
# from .plugin import RetrievalPlugin

THRESH = 1
CHUNK_BUDGET = 2000

async def suppmat_query(
    datastore,
    json_db,
    query,
    IDs,
    tags=["supporting material"],
    chunksizes=[1000],
    top_k=80,
    clean_results=True,
):
    respd = [rec for rec in json_db if rec["id"] in IDs]
    include_docs = [
        ref["full_text"]["document_id"]
        for respd_ in respd
        for ref in respd_.get("refs", {})
        if ref["full_text"]["available"]
    ]
    if include_docs:
        return await db_query(
            datastore,
            query,
            include_docs=include_docs,
            chunksize=chunksizes,
            top_k=top_k,
            clean_results=clean_results,
            tags=tags,
        )
    else:
        return SimpleNamespace(results=[])


@delayedretry(rethrow_final_error=True, include_errors=[ConnectionError])
async def db_query(
    datastore,
    query,
    exclude_docs=[],
    include_docs=[],
    tags=[],
    chunksize=[800, 1000],
    top_k=80,
    verbose=0,
    clean_results=True,
):
    if verbose > 0:
        print(f"{query}\n")
    filt_in = DocumentMetadataFilter(
        tag="|".join(tags),
        chunksize="|".join(str(cs) for cs in chunksize),
        document_id="|".join(include_docs),
    )
    filt_out = DocumentMetadataFilter(document_id="|".join(exclude_docs))
    st = time.perf_counter()
    q = Query(query=query, top_k=top_k, filter_in=filt_in, filter_out=filt_out)
    print(f"1: {time.perf_counter() - st}")
    out = (await datastore.query([q]))[0]
    if verbose > 0:
        print([r.metadata.tag for r in out.results])
        print([r.chunksize for r in out.results])
    st = time.perf_counter()
    if clean_results:
        out.results = [r for r in out.results if len(r.text.split()) > 0]
        if out.results:
            out.results = await remove_duplicate_results(out.results, verbose)
    print(f"2: {time.perf_counter() - st}")
    return out


async def remove_duplicate_results(results, verbose=0):
    embs = np.vstack([r.embedding for r in results])
    cc = np.triu(np.corrcoef(embs), k=1)
    dropidx = []
    for i, j in zip(*np.nonzero(cc > 0.99)):
        if i not in dropidx:
            dropidx.append(j)
    if verbose > 0:
        print(embs.shape)
        print(dropidx)
    return [r for i, r in enumerate(results) if i not in dropidx]


def format_query_results(results, thresh=THRESH, chunk_budget=CHUNK_BUDGET):
    res = sorted((r for r in results if r.score < thresh), key=lambda x: x.score)
    totcost = 0
    res_ = []
    for r in res:
        if totcost <= chunk_budget:
            res_.append(r)
            totcost += int(r.chunksize)
        else:
            break
    res = res_
    respd = defaultdict(lambda: {"chunks": []})
    for r in res:
        respd[r.metadata.document_id]["document ID"] = r.metadata.document_id
        respd[r.metadata.document_id]["title"] = r.metadata.title
        respd[r.metadata.document_id]["author"] = r.metadata.author
        respd[r.metadata.document_id]["abstract"] = r.metadata.abstract
        respd[r.metadata.document_id]["published_in"] = r.metadata.published_in
        respd[r.metadata.document_id]["year"] = r.metadata.year
        respd[r.metadata.document_id]["url"] = r.metadata.url
        respd[r.metadata.document_id]["chunks"].append(r.text)
    resp = ""
    for filename, filed in respd.items():
        for key, val in filed.items():
            resp += f"{key}: {val}\n"
        for i, chunk in enumerate(filed["chunks"]):
            resp += f"chunk_{i :03d}: {chunk}\n"
        resp += "\n\n"
    return respd
