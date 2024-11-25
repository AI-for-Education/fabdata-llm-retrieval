import json
from typing import Dict, List

from fdllm.tooluse import Tool, ToolParam, ToolItem
from pydantic import Field
from fuzzywuzzy import fuzz, process

from .datastore.datastore import DataStore
from .helpers import db_query, suppmat_query, format_query_results


class QueryCatalogue(Tool):
    def __init__(
        self,
        tags: List[str] = ["guides", "research", "reviews"],
        chunksizes: List[int] = [200, 400, 600, 800, 1000],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.params["tags"].items.enum = tags
        self.params["chunksize"].items.enum = chunksizes

    name = "query_catalogue"
    description = (
        f"Query chunks of text from the catalogue by description."
        " You can also filter by tag, document_id, and chunksize, or you can choose"
        " to exclude certain document_ids from the search."
    )

    params = {
        "query": ToolParam(
            type="string",
            description=(
                "String to query for related text chunks in the catalogue."
                " The more descriptive it is, the more accurate the results will be."
            ),
            required=True,
        ),
        "tags": ToolParam(
            type="array",
            items=ToolItem(type="string"),
            description="Tags to filter results by. Only results with these tags will be included.",
            required=True,
        ),
        "exclude_docs": ToolParam(
            type="array",
            items=ToolItem(type="string"),
            description=(
                "Exclude these document IDs from the search."
                " This can be used whenever you need to find something that hasn't already been found by the same query."
            ),
            default=[],
        ),
        "include_docs": ToolParam(
            type="array",
            items=ToolItem(type="string"),
            description="Document IDs to filter results by. Only document with these IDs will be included",
            default=[],
        ),
        "chunksize": ToolParam(
            type="array",
            items=ToolItem(type="integer"),
            description=(
                "Chunksizes to filter results by. Only results of these chunksizes will be included."
                " Smaller chunks are better for finding a broad array of different documents, wheareas larger chunks"
                " are better for digging into the details of a particular document or set of documents."
            ),
            required=True,
        ),
    }

    datastore: DataStore
    top_k: int = 80
    clean_results: bool = True
    verbose: int = 0

    def execute(self, **params):
        raise NotImplementedError()

    async def aexecute(self, **params):
        out = await db_query(
            datastore=self.datastore,
            top_k=self.top_k,
            clean_results=self.clean_results,
            **params,
        )
        return json.dumps(format_query_results(out.results))


class GetContents(Tool):
    name = "get_contents"
    description = (
        "Return the contents of the catalogue."
        " This returns a lists of topics, rather than a list of documents."
    )
    params = {}
    json_contents: Dict

    def execute(self, **params):
        return super().execute(**params)

    async def aexecute(self, **params):
        return json.dumps(self.json_contents)


class GetReferences(Tool):
    name = "get_references"
    description = "Return the references list from a document by its ID"
    params = {
        "ID": ToolParam(
            type="string", description="ID of document to return", required=True
        )
    }
    json_database: Dict | List[Dict]

    def execute(self, **params):
        return super().execute(**params)

    async def aexecute(self, **params):
        respd = [rec for rec in self.json_database if rec["id"] == params["ID"]][0]
        resp = ""
        resp += f"filename: {respd['filename']}\n"
        resp += f"document ID: {respd['id']}\n"
        resp += f"url: {respd.get('url', None)}\n"
        resp += f"references:\n{json.dumps(respd.get('refs', {}), indent=4)}"
        return resp


class QuerySuppMat(Tool):
    name = "query_supporting_material"
    description = "Query chunks of text from a document's supporting material"
    params = {
        "query": ToolParam(
            type="string",
            description=(
                "String to query for related text chunks in the supporting material."
                " Try to summarise of the chunks of interest from the document in order to"
                " find the most relevant supporting material chunks."
            ),
            required=True,
        ),
        "IDs": ToolParam(
            type="array",
            items=ToolItem(type="string"),
            description="IDs of documents to get supporting materials",
            required=True,
        ),
    }
    datastore: DataStore
    json_database: Dict | List[Dict]
    top_k: int = 80
    clean_results: bool = True
    verbose: int = 0
    tags: List[str] = Field(default_factory=lambda: ["supporting material"])
    chunksizes: List[int] = Field(default_factory=lambda: [1000])

    def execute(self, **params):
        return super().execute(**params)

    async def aexecute(self, **params):
        out = await suppmat_query(
            datastore=self.datastore,
            json_db=self.json_database,
            top_k=self.top_k,
            clean_results=self.clean_results,
            tags=self.tags,
            chunksizes=self.chunksizes,
            **params,
        )
        return json.dumps(format_query_results(out))


class FullText(Tool):
    name = "full_text"
    description = "Return the full text of a document by its ID"
    params = {
        "ID": ToolParam(
            type="string", description="ID of document to return", required=True
        )
    }
    json_database: Dict | List[Dict]

    def execute(self, **params):
        return super().execute(**params)

    async def aexecute(self, **params):
        ft = [rec for rec in self.json_database if rec["id"] == params["ID"]][0]
        if len(ft["text"]) > 60000:
            ft["text"] = ft["text"][:60000] + "..."
            
        return json.dumps(ft)

