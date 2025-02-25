from typing import Literal
from collections import defaultdict
from types import SimpleNamespace
import time
import json

import numpy as np
from fdllm import get_caller, LLMMessage
from fdllm.llmtypes import LLMCaller
from fdllm.chat import ChatController
from fdllm.decorators import delayedretry
from redis.exceptions import ConnectionError
from pydantic import BaseModel

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
    chunksizes=["1000"],
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
    chunksize=["800", "1000"],
    top_k=80,
    verbose=0,
    clean_results=True,
    **kwargs,
):
    if verbose > 0:
        print(f"{query}\n")
    ## with tags
    filt_in = DocumentMetadataFilter(
        tag="|".join(tags),
        chunksize="|".join(cs for cs in chunksize),
        document_id="|".join(include_docs),
    )
    ## without tags
    filt_in_notag = DocumentMetadataFilter(
        chunksize="|".join(cs for cs in chunksize),
        document_id="|".join(include_docs),
    )
    filt_out = DocumentMetadataFilter(document_id="|".join(exclude_docs))
    st = time.perf_counter()
    q = Query(query=query, top_k=top_k, filter_in=filt_in, filter_out=filt_out)
    q_notag = Query(query=query, top_k=top_k, filter_in=filt_in_notag, filter_out=filt_out)
    print(f"1: {time.perf_counter() - st}")
    out = (await datastore.query([q]))[0]
    out_notag = (await datastore.query([q_notag]))[0]
    if verbose > 0:
        print([r.metadata.tag for r in out.results])
        print([r.chunksize for r in out.results])
    out.results = [*out.results, *out_notag.results]
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



class RelevanceFormat(BaseModel):
    class DocRelevance(BaseModel):
        class ChunkRelevance(BaseModel):
            query_relevance: Literal["very_low", "low", "medium", "high", "very_high"]
            intention_relevance: Literal["very_low", "low", "medium", "high", "very_high"]
        
        document_id: str
        overall_query_relevance: Literal["very_low", "low", "medium", "high", "very_high"]
        overall_intention_relevance: Literal["very_low", "low", "medium", "high", "very_high"]
        chunk_relevance: list[ChunkRelevance]
    
    document_relevance: list[DocRelevance]
    
async def results_relevance(results: dict, query: str, intention: str, caller: LLMCaller):
    instruction = (
        "Below are some chunks of text that were automatically extracted"
        " from a catolugue of documents based on a query string."
        " The query string itself was generated to help a user answer a specific"
        " question, which we call the 'intention'."
        " The automatic process has a tendency to pick up irrelevant chunks"
        " and documents."
        " Please rate the relevance of each chunk to the query and to the intention and then"
        " rate the relevance of the document that they appeared in."
        "\n\n"
        "<<chunks>>"
        "\n\n"
        f"{json.dumps(results)}"
        "\n\n"
        "<<query string>>"
        "\n\n"
        f"{query}"
        "\n\n"
        "<<intention>>"
        f"{intention}"
    )
    msg = LLMMessage(Role="user", Message=instruction)
    out = await caller.acall(msg, max_tokens=None, response_schema=RelevanceFormat, temperature=0)
    formatted_out = RelevanceFormat.model_validate_json(out.Message)
    
    outres = {}
    for docrel in formatted_out.document_relevance:
        if docrel.overall_intention_relevance in ["high", "very_high"]:
            candoc = results[docrel.document_id]
            candoc["chunks"] = [
                chunk for chunk, chunkrel in zip(candoc["chunks"], docrel.chunk_relevance)
                if chunkrel.intention_relevance in ["high", "very_high"]
            ]
            if candoc["chunks"]:
                outres[docrel.document_id] = candoc

    return outres