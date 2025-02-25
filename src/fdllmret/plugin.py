from fdllm import get_caller
from fdllm.tooluse import ToolUsePlugin
from typing import Optional, Union, List
import os

from .datastore.factory import get_datastore
from .tools import *
from .helpers.encoding import DocsetEncoding
from .helpers.query import THRESH, CHUNK_BUDGET


async def retrieval_plugin(
    dbhost: Optional[str] = None,
    dbport: Optional[int] = None,
    dbssl: Optional[bool] = None,
    dbauth: Optional[str] = None,
    chunksizes: Optional[Union[str, List[str]]] = None,
    thresh: float = THRESH,
    chunk_budget: int = CHUNK_BUDGET,
    relevance_model: Optional[str] = None,
):
    client_kwargs = {}
    if dbhost is not None:
        client_kwargs["redis_host"] = dbhost
    if dbport is not None:
        client_kwargs["redis_port"] = dbport
    if dbssl is not None:
        client_kwargs["redis_ssl"] = dbssl
    if dbauth is not None:
        client_kwargs["redis_password"] = dbauth

    datastore = await get_datastore(**client_kwargs)
    docenc = await DocsetEncoding.from_datastore(datastore)

    if chunksizes is None:
        chunksizes = [str(cs) for cs in docenc.chunk_sizes]
    else:
        if isinstance(chunksizes, str):
            chunksizes = [chunksizes]
        if not set(chunksizes).issubset(set(docenc.chunk_sizes)):
            raise ValueError("chunksize must be a subset of docenc.docembs.chunk_sizes")

    if relevance_model is not None:
        relevance_caller = get_caller(relevance_model)
    else:
        relevance_caller = None
    plugin = RetrievalPlugin(
        datastore=datastore,
        json_contents=docenc.contents,
        json_database=docenc.jsondata,
        chunksizes=chunksizes,
        tags=docenc.tags,
        supp_tags=docenc.supp_tags,
        thresh=thresh,
        chunk_budget=chunk_budget,
        relevance_caller=relevance_caller,
    )

    return plugin, datastore


class RetrievalPlugin(ToolUsePlugin):
    def __init__(
        self,
        datastore,
        json_contents,
        json_database,
        chunksizes,
        tags,
        supp_tags,
        thresh=THRESH,
        chunk_budget=CHUNK_BUDGET,
        relevance_caller=None,
    ):
        tools = [
            QueryCatalogue(
                datastore=datastore,
                tags=tags,
                chunksizes=chunksizes,
                thresh=thresh,
                chunk_budget=chunk_budget,
                relevance_caller=relevance_caller,
            ),
            GetReferences(json_database=json_database),
            FullText(json_database=json_database),
        ]
        if json_contents:
            tools.append(GetContents(json_contents=json_contents))
        if supp_tags:
            tools.append(
                QuerySuppMat(
                    datastore=datastore,
                    json_database=json_database,
                    tags=supp_tags,
                    chunksizes=chunksizes[-1:],
                    thresh=thresh,
                    chunk_budget=chunk_budget,
                ),
            )
        super().__init__(Tools=tools)
