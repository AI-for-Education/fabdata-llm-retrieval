from typing import List, Literal, Union
import os
from types import SimpleNamespace
import uuid
import json
import argparse
import asyncio
from pathlib import Path
import pickle
from collections import defaultdict

import yaml
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

from ..models.models import Document, DocumentMetadata, Optional
from ..services.chunks import get_document_chunks, CHUNK_SIZE
from .extraction import process_folder, process_path, load_jsondata, save_jsondata
from .analysis import auto_tag_generate


PATHTYPE = Union[os.PathLike, str]


class DocsetEncoding:
    def __init__(
        self,
        folder: PATHTYPE,
        cachedir: PATHTYPE,
        extract_refs: bool = False,
        download_refs: bool = False,
        n_jobs: int = -1,
        pdfengine: Literal["fitz", "pypdf"] = "fitz",
        exts: list[str] = [".pdf"],
        embeddingsengine: Literal["openai_ada-002"] = "openai_ada-002",
        extraction_model: str = "gpt-4-1106-preview",
        custom_models_config: Optional[PATHTYPE] = None,
        chunk_size: Union[int, List[int]] = [200, 400, 600, 800, 1000],
        auto_tag: bool = False,
        auto_tag_parameters: dict = {"chunk_size": 800},
        create_cache=True,
    ):
        self._cachedir = process_path(cachedir)
        if create_cache:
            self._cachedir.mkdir(exist_ok=True, parents=True)
        self.jsondata = None
        self.contents = None
        self.enc = None
        self.docembs = None
        self.supp_tags = None
        self.config = {
            "general": dict(
                cachedir=cachedir, custom_models_config=custom_models_config
            ),
            "extraction": dict(
                folder=folder,
                extract_refs=extract_refs,
                download_refs=download_refs,
                n_jobs=n_jobs,
                pdfengine=pdfengine,
                extraction_model=extraction_model,
                exts=exts,
            ),
            "encoding": dict(
                embeddingsengine=embeddingsengine,
                chunk_size=chunk_size,
            ),
            "tagging": dict(
                auto_tag=auto_tag,
                auto_tag_parameters=auto_tag_parameters,
            ),
        }
        if not self._assert_cache_safe():
            raise ValueError("Cache settings don't match object settings")
        if create_cache:
            with open(self.configfile, "w") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)

    def __getitem__(self, docid):
        doc = self._document(docid)
        chunks = {doc.id: self.enc[doc.id]}
        return doc, chunks

    def __len__(self):
        return len(self.jsondata)

    def _document(self, docid):
        if isinstance(docid, int):
            item = self.jsondata[docid]
        elif isinstance(docid, str):
            item = [it for it in self.jsondata if it["id"] == docid]
            if not item:
                raise IndexError(f"Invalid index")
            elif len(item) > 1:
                raise IndexError(f"{docid} must be unique")
        return create_document(item)

    def extract(self):
        if not self._assert_cache_safe():
            raise ValueError("Cache settings don't match object settings")
        self._process_folder(**self.config["extraction"])

    def encode(self, verbose=0):
        if not self._assert_cache_safe():
            raise ValueError("Cache settings don't match object settings")
        if self.jsondata is None:
            self.extract(verbose=verbose)
        if verbose > 0:
            print("Encoding chunks")
        self.enc = encode_documents_json(
            self.jsondata,
            **self.config["encoding"],
            cachedir=self._cachedir,
            verbose=verbose,
        )
        self.docembs = DocsetEmbeddings(self.enc, self.config["encoding"]["chunk_size"])

    def auto_tag(
        self,
        chunk_size,
        n_clusters=10,
        clusterer=KMeans,
        cluster_kwargs={"n_init": 10},
        matrix_kwargs={},
        n_top=3,
        overwrite=False,
        **auto_tag_kwargs,
    ):
        if self.tags is not None and not overwrite:
            return

        cluster_labs, scores = self.docembs.cluster(
            chunk_size,
            n_clusters=n_clusters,
            clusterer=clusterer,
            cluster_kwargs=cluster_kwargs,
            **matrix_kwargs,
        )
        topnchunks = self.topn_cluster_chunks(chunk_size, n_top, cluster_labs, scores)
        tags, flattags = auto_tag_generate(
            cluster_labs, scores, topnchunks, chunk_size, **auto_tag_kwargs
        )
        self._apply_tags(chunk_size, cluster_labs, tags)
        self.to_cache()

    def topn_cluster_chunks(
        self,
        chunk_size,
        n_top,
        clusters,
        scores,
        n_start=0,
        id_strings=None,
    ) -> dict:

        unq_clusters = np.unique(clusters)
        n_clusters = len(unq_clusters)
        if id_strings is None:
            id_strings = [str(uuid.uuid4()) for _ in range(n_clusters)]
        else:
            if not isinstance(id_strings, list) or len(id_strings) != n_clusters:
                raise ValueError(
                    "Number of id strings if provided must match number of clusters"
                )
        docembs = self.docembs
        topnchunks = {}
        for lab, id_str in zip(unq_clusters, id_strings):
            labidx = np.flatnonzero(clusters == lab)
            subscores = scores[labidx]
            sortscores = np.argsort(subscores)[::-1][n_start:]
            sortidx = labidx[sortscores]
            encidxs = [docembs.inverseidx[chunk_size][i] for i in sortidx]
            gotdoc = []
            topenc = []
            for id_, i in encidxs:
                if id_ not in gotdoc:
                    gotdoc.append(id_)
                    topenc.append(self.enc[id_][i])
                    if len(topenc) == n_top:
                        break
            topnchunks[id_str] = [te.text for te in topenc]
        return topnchunks

    def _apply_tags(self, chunk_size, cluster_labs, tags):
        for c in np.unique(cluster_labs):
            tagstr = ",".join(tags[c])
            cidx = np.flatnonzero(cluster_labs == c)
            cdocs = [
                didx
                for i, (didx, _) in enumerate(self.docembs.inverseidx[chunk_size])
                if i in cidx
            ]
            for doci in set(cdocs):
                for chunk in self.enc[doci]:
                    chunk.metadata.tag = tagstr
                for doc in self.jsondata:
                    if doc["id"] == doci:
                        doc["tag"] = tagstr

    @property
    def configfile(self):
        return self._cachedir / "config.yml"

    @property
    def jsondatafile(self):
        return self._cachedir / "jsondata"

    @property
    def contentsfile(self):
        return self._cachedir / "contents.json"

    @property
    def tags(self):
        tags = set()
        for doc in self.jsondata:
            tags.update(doc["tag"].split(","))
        if not any(tg for tg in tags):
            tags = None
        else:
            tags = sorted(tags)
        return tags

    @property
    def chunk_sizes(self):
        cs = self.config["encoding"]["chunk_size"]
        if not isinstance(cs, list):
            cs = [cs]
        return cs

    def picklefile(self, chunk_size):
        pklbase = self._cachedir
        return pklbase / f"{pklbase.name}_encoded_{chunk_size}.pkl"

    @classmethod
    def from_config(
        cls,
        config_file: Optional[PATHTYPE] = None,
        config_dict: Optional[dict] = None,
        extract=True,
        encode=True,
        create_cache=True,
    ):
        if config_file is None and config_dict is None:
            raise ValueError("Must supply one of config_file or config_dict")
        if all(cf is not None for cf in [config_file, config_dict]):
            raise ValueError("Must supply only one of config_file or config_dict")
        if config_file is not None:
            config_file = Path(config_file)
            if config_file.exists():
                with open(config_file) as f:
                    cfg = yaml.safe_load(f)
            else:
                raise OSError(f"{config_file} doesn't exist")
        if config_dict is not None:
            cfg = config_dict
        out = cls(
            **cfg["general"],
            **cfg["extraction"],
            **cfg["encoding"],
            **cfg["tagging"],
            create_cache=create_cache,
        )
        if extract:
            out.extract()
        if encode:
            out.encode(verbose=True)
            if out.config["tagging"]["auto_tag"]:
                out.auto_tag(**out.config["tagging"]["auto_tag_parameters"])
        return out

    @classmethod
    def from_cache(cls, cachedir: PATHTYPE):
        cachedir = Path(cachedir)
        cfgfile = cachedir / "config.yml"
        return cls.from_config(cfgfile)

    @classmethod
    async def from_datastore(cls, datastore):
        config = (await datastore.client.json().get("config", "$"))[0]
        out = DocsetEncoding.from_config(
            config_dict=config, extract=False, encode=False, create_cache=False
        )
        out.jsondata = (await datastore.client.json().get("fulldb", "$"))[0]
        out.contents = (await datastore.client.json().get("contents", "$"))[0]
        return out

    def to_cache(self):
        with open(self.contentsfile, "w") as f:
            json.dump(self.contents, f)
        save_jsondata(self.jsondata, self.jsondatafile)
        for chunk_size in self.docembs.chunk_sizes:
            enc = {
                key: [chunk for chunk in val if f"_{chunk_size}_" in chunk.id]
                for key, val in self.enc.items()
            }
            with open(self.picklefile(chunk_size), "wb") as f:
                pickle.dump(enc, f)

    def _process_folder(self, **kwargs):
        jsondata, contents = process_folder(self, **kwargs)
        save_jsondata(jsondata, self.jsondatafile)
        with open(self.contentsfile, "w") as f:
            json.dump(contents, f, indent=4)
        self.jsondata = jsondata
        self.contents = contents

    def _assert_cache_safe(self):
        if self.configfile.exists():
            with open(self.configfile) as f:
                cachecfg = yaml.safe_load(f)
            return cachecfg == self.config
        else:
            return True


class DocsetEmbeddings:
    def __init__(self, encodings, chunk_size):
        self.inverseidx = None
        self.docidx = None
        self._extract_embeddings(encodings, chunk_size)
        self.tsne = DocsetTSNE(self)

    def matrix(
        self, chunk_size=None, doc_agg=False, with_nchunks=False, normalize=False
    ):
        if chunk_size is None:
            if doc_agg:
                emb = self._alldocembmat
            else:
                emb = self._allembmat
        else:
            if chunk_size not in self.chunk_sizes:
                raise ValueError("Invalid chunk size")
            else:
                if doc_agg:
                    emb = self._docembmats[chunk_size]
                else:
                    emb = self._embmats[chunk_size]
        if not with_nchunks and not doc_agg:
            emb = emb[:, 2:]

        if normalize:
            emb = emb / np.linalg.norm(axis=0, keepdims=True)

        return emb

    def cluster(
        self,
        chunk_size,
        n_clusters=10,
        clusterer=KMeans,
        cluster_kwargs={},
        **matrix_kwargs,
    ):
        X = self.matrix(chunk_size, **matrix_kwargs)
        cl = clusterer(n_clusters=n_clusters, **cluster_kwargs)
        cluster_labs = cl.fit_predict(X)
        scores = silhouette_samples(X, cluster_labs)
        return cluster_labs, scores

    def _extract_embeddings(self, encodings, chunk_size):
        if not isinstance(chunk_size, list):
            chunk_size = [chunk_size]
        self.chunk_sizes = chunk_size
        embmats = {}
        docembmats = {}
        docidx = defaultdict(dict)
        inverseidx = {}
        for cs in chunk_size:
            emblist = []
            docemblist = []
            invidxlist = []
            idxstep = 0
            for id, chunks in encodings.items():
                emblistdoc = []
                cnt = 0
                for i, chunk in enumerate(chunks):
                    if f"_{cs}_" in chunk.id:
                        cnt += 1
                        emblistdoc.append(chunk.embedding)
                        invidxlist.append((id, i))
                for i in range(cnt):
                    emblistdoc[i] = np.array([*[i / cnt, cnt], *emblistdoc[i]])
                docemblist.append(np.mean(emblistdoc, axis=0)[2:])
                emblist.extend(emblistdoc)
                docidx[cs][id] = np.arange(idxstep, idxstep + cnt)
                idxstep += cnt
            embmats[cs] = np.vstack(emblist)
            docembmats[cs] = np.vstack(docemblist)
            inverseidx[cs] = invidxlist

        allembmat = np.vstack(list(embmats.values()))
        alldocembmat = np.vstack(list(docembmats.values()))

        self._embmats = embmats
        self._docembmats = docembmats
        self._allembmat = allembmat
        self._alldocembmat = alldocembmat
        self.docidx = docidx
        self.inverseidx = inverseidx


class DocsetTSNE:
    def __init__(self, embeddings: DocsetEmbeddings, random_state: int = 92373263):
        self.embeddings = embeddings
        self._random_state = random_state
        self._tsne_vecs = {}
        self._doc_tsne_vecs = {}
        self._all_tsne_vecs = None
        self._all_doc_tsne_vecs = None

    def __call__(self, *args, **kwargs):
        return self.vectors(*args, **kwargs)

    @property
    def chunk_sizes(self):
        return self.embeddings.chunk_sizes

    def vectors(self, chunk_size=None, doc_agg=False):
        if chunk_size is None:
            if doc_agg:
                if self._all_doc_tsne_vecs is not None:
                    return self._all_doc_tsne_vecs
                else:
                    self._all_doc_tsne_vecs = self._tsne(chunk_size=None, doc_agg=True)
                    return self._all_doc_tsne_vecs
            else:
                if self._all_tsne_vecs is not None:
                    return self._all_tsne_vecs
                else:
                    self._all_tsne_vecs = self._tsne(chunk_size=None, doc_agg=False)
                    return self._all_tsne_vecs
        else:
            if doc_agg:
                if chunk_size in self._doc_tsne_vecs:
                    return self._doc_tsne_vecs[chunk_size]
                else:
                    self._doc_tsne_vecs[chunk_size] = self._tsne(
                        chunk_size=chunk_size, doc_agg=True
                    )
                    return self._doc_tsne_vecs[chunk_size]
            else:
                if chunk_size in self._tsne_vecs:
                    return self._tsne_vecs[chunk_size]
                else:
                    self._tsne_vecs[chunk_size] = self._tsne(
                        chunk_size=chunk_size, doc_agg=False
                    )
                    return self._tsne_vecs[chunk_size]

    def __call__(self, **kwargs):
        return self._tsne_embeddings(**kwargs)

    def _tsne(self, *matrix_args, **matrix_kwargs):
        embs = self.embeddings.matrix(*matrix_args, **matrix_kwargs)
        return TSNE(random_state=self._random_state).fit_transform(embs)

    def plot(
        self, chunk_sizes=None, doc_agg=False, ax=None, figsize=(8, 8), **sct_kwargs
    ):
        sct_kwargs_defaults = {"s": 6}
        sct_kwargs = {**sct_kwargs_defaults, **sct_kwargs}
        if not isinstance(chunk_sizes, (list, str)):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
                fig.tight_layout()
            vectors = self.vectors(chunk_size=chunk_sizes, doc_agg=doc_agg)
            ax.scatter(*vectors.T, **sct_kwargs)
        else:
            if isinstance(chunk_sizes, str):
                if chunk_sizes == "all":
                    chunk_sizes = self.chunk_sizes
                else:
                    raise ValueError(
                        f"chunk_sizes should be one of {None}, {List[int]}, 'all'"
                    )
            if ax is None:
                nchunks = len(chunk_sizes)
                fig, ax = plt.subplots(
                    ncols=nchunks,
                    figsize=((figsize[0] * nchunks) // 2, figsize[1] // 2),
                )
                fig.tight_layout()
            for ax_, cs in zip(ax, chunk_sizes):
                self.plot(chunk_sizes=cs, doc_agg=doc_agg, ax=ax_, **sct_kwargs)

        return ax


def encode_documents_json(
    documents_json,
    chunk_size,
    cachedir=None,
    embeddingsengine="openai_ada-002",
    verbose=0,
):
    documents = [create_document(docjson) for docjson in documents_json]
    fullenc = defaultdict(list)
    for chunk_size_ in chunk_size:
        if cachedir is not None:
            pklbase = Path(cachedir)
            pklfile = pklbase / f"{pklbase.name}_encoded_{chunk_size_}.pkl"
        if cachedir is None or not pklfile.exists():
            enc = encode_documents(
                documents=documents, chunk_size=[chunk_size_], verbose=verbose
            )
            if cachedir is not None:
                pklfile.parent.mkdir(exist_ok=True, parents=True)
                with open(pklfile, "wb") as f:
                    pickle.dump(enc, f)
        else:
            with open(pklfile, "rb") as f:
                enc = pickle.load(f)
        for key, val in enc.items():
            fullenc[key].extend(val)

    return dict(fullenc)


def create_document(docjson):
    id = docjson.get("id", str(uuid.uuid4()))
    text = docjson.get("text", None)
    metadata = DocumentMetadata(
        source=docjson.get("source"),
        source_id=docjson.get("source_id"),
        filename=docjson.get("filename"),
        tag=docjson.get("tag"),
        **docjson.get("metadata", {}),
    )

    # create a document object with the id or a random id, text and metadata
    return Document(
        id=id,
        text=text,
        metadata=metadata,
    )


def encode_documents(
    documents: List[Document], chunk_size: List[int], verbose: int = 0
):
    docchunks = defaultdict(list)
    for chunk_size_ in chunk_size:
        if verbose > 0:
            print(f"chunk size: {chunk_size_}")
        chunks = get_document_chunks(documents, chunk_size_, verbose)
        for key, val in chunks.items():
            docchunks[key].extend(val)
    return docchunks
