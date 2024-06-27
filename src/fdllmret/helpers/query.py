from collections import defaultdict
from types import SimpleNamespace

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
    q = Query(query=query, top_k=top_k, filter_in=filt_in, filter_out=filt_out)
    out = (await datastore.query([q]))[0]
    if verbose > 0:
        print([r.metadata.tag for r in out.results])
        print([r.chunksize for r in out.results])
    if clean_results:
        out.results = [r for r in out.results if len(r.text.split()) > 0]
        if out.results:
            out.results = await remove_duplicate_results(out.results, verbose)
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


def format_query_results(results):
    res = sorted((r for r in results if r.score < THRESH), key=lambda x: x.score)
    totcost = 0
    res_ = []
    for r in res:
        if totcost <= CHUNK_BUDGET:
            res_.append(r)
            totcost += int(r.chunksize)
        else:
            break
    res = res_
    respd = defaultdict(lambda: {"id": None, "chunks": []})
    for r in res:
        respd[r.metadata.filename]["id"] = r.metadata.document_id
        respd[r.metadata.filename]["url"] = r.metadata.url
        respd[r.metadata.filename]["chunks"].append(r.text)
    resp = ""
    for filename, filed in respd.items():
        resp += f"filename: {filename}\n"
        resp += f"document ID: {filed['id']}\n"
        resp += f"url: {filed['url']}\n"
        for i, chunk in enumerate(filed["chunks"]):
            resp += f"chunk_{i :03d}: {chunk}\n"
        resp += "\n\n"
    return respd
