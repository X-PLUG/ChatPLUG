# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import io
import os
from os.path import expanduser
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import requests


ENDPOINT = "https://huggingface.co"


class RepoObj:
    """
    HuggingFace git-based system, data structure that represents a file belonging to the current user.
    """

    def __init__(self, filename: str, lastModified: str, commit: str, size: int, **kwargs):
        self.filename = filename
        self.lastModified = lastModified
        self.commit = commit
        self.size = size


class S3Obj:
    """
    HuggingFace S3-based system, data structure that represents a file belonging to the current user.
    """

    def __init__(self, filename: str, LastModified: str, ETag: str, Size: int, **kwargs):
        self.filename = filename
        self.LastModified = LastModified
        self.ETag = ETag
        self.Size = Size


class PresignedUrl:
    def __init__(self, write: str, access: str, type: str, **kwargs):
        self.write = write
        self.access = access
        self.type = type  # mime-type to send to S3.


class ModelSibling:
    """
    Data structure that represents a public file inside a model, accessible from huggingface.co
    """

    def __init__(self, rfilename: str, **kwargs):
        self.rfilename = rfilename  # filename relative to the model root
        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelInfo:
    """
    Info about a public model accessible from huggingface.co
    """

    def __init__(
        self,
        modelId: Optional[str] = None,  # id of model
        author: Optional[str] = None,
        downloads: Optional[int] = None,
        tags: List[str] = [],
        pipeline_tag: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,  # list of files that constitute the model
        **kwargs
    ):
        self.modelId = modelId
        self.author = author
        self.downloads = downloads
        self.tags = tags
        self.pipeline_tag = pipeline_tag
        self.siblings = [ModelSibling(**x) for x in siblings] if siblings is not None else None
        for k, v in kwargs.items():
            setattr(self, k, v)


class HfApi:
    ALLOWED_S3_FILE_TYPES = ["datasets", "metrics"]

    def __init__(self, endpoint=None):
        self.endpoint = endpoint if endpoint is not None else ENDPOINT

    def login(self, username: str, password: str) -> str:
        """
        Call HF API to sign in a user and get a token if credentials are valid.

        Outputs: token if credentials are valid

        Throws: requests.exceptions.HTTPError if credentials are invalid
        """
        path = "{}/api/login".format(self.endpoint)
        r = requests.post(path, json={"username": username, "password": password})
        r.raise_for_status()
        d = r.json()
        return d["token"]

    def whoami(self, token: str) -> Tuple[str, List[str]]:
        """
        Call HF API to know "whoami"
        """
        path = "{}/api/whoami".format(self.endpoint)
        r = requests.get(path, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()
        d = r.json()
        return d["user"], d["orgs"]

    def logout(self, token: str) -> None:
        """
        Call HF API to log out.
        """
        path = "{}/api/logout".format(self.endpoint)
        r = requests.post(path, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()

    def presign(self, token: str, filetype: str, filename: str, organization: Optional[str] = None) -> PresignedUrl:
        """
        HuggingFace S3-based system, used for datasets and metrics.

        Call HF API to get a presigned url to upload `filename` to S3.
        """
        assert filetype in self.ALLOWED_S3_FILE_TYPES, f"Please specify filetype from {self.ALLOWED_S3_FILE_TYPES}"
        path = f"{self.endpoint}/api/{filetype}/presign"
        r = requests.post(
            path,
            headers={"authorization": "Bearer {}".format(token)},
            json={"filename": filename, "organization": organization},
        )
        r.raise_for_status()
        d = r.json()
        return PresignedUrl(**d)

    def presign_and_upload(
        self, token: str, filetype: str, filename: str, filepath: str, organization: Optional[str] = None
    ) -> str:
        """
        HuggingFace S3-based system, used for datasets and metrics.

        Get a presigned url, then upload file to S3.

        Outputs: url: Read-only url for the stored file on S3.
        """
        assert filetype in self.ALLOWED_S3_FILE_TYPES, f"Please specify filetype from {self.ALLOWED_S3_FILE_TYPES}"
        urls = self.presign(token, filetype=filetype, filename=filename, organization=organization)
        # streaming upload:
        # https://2.python-requests.org/en/master/user/advanced/#streaming-uploads
        #
        # Even though we presign with the correct content-type,
        # the client still has to specify it when uploading the file.
        with open(filepath, "rb") as f:
            pf = TqdmProgressFileReader(f)
            data = f if pf.total_size > 0 else ""

            r = requests.put(urls.write, data=data, headers={"content-type": urls.type})
            r.raise_for_status()
            pf.close()
        return urls.access

    def list_objs(self, token: str, filetype: str, organization: Optional[str] = None) -> List[S3Obj]:
        """
        HuggingFace S3-based system, used for datasets and metrics.

        Call HF API to list all stored files for user (or one of their organizations).
        """
        assert filetype in self.ALLOWED_S3_FILE_TYPES, f"Please specify filetype from {self.ALLOWED_S3_FILE_TYPES}"
        path = "{}/api/{}/listObjs".format(self.endpoint, filetype)
        params = {"organization": organization} if organization is not None else None
        r = requests.get(path, params=params, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()
        d = r.json()
        return [S3Obj(**x) for x in d]

    def delete_obj(self, token: str, filetype: str, filename: str, organization: Optional[str] = None):
        """
        HuggingFace S3-based system, used for datasets and metrics.

        Call HF API to delete a file stored by user
        """
        assert filetype in self.ALLOWED_S3_FILE_TYPES, f"Please specify filetype from {self.ALLOWED_S3_FILE_TYPES}"
        path = "{}/api/{}/deleteObj".format(self.endpoint, filetype)
        r = requests.delete(
            path,
            headers={"authorization": "Bearer {}".format(token)},
            json={"filename": filename, "organization": organization},
        )
        r.raise_for_status()

    def model_list(self) -> List[ModelInfo]:
        """
        Get the public list of all the models on huggingface.co
        """
        path = "{}/api/models".format(self.endpoint)
        r = requests.get(path)
        r.raise_for_status()
        d = r.json()
        return [ModelInfo(**x) for x in d]

    def list_repos_objs(self, token: str, organization: Optional[str] = None) -> List[RepoObj]:
        """
        HuggingFace git-based system, used for models.

        Call HF API to list all stored files for user (or one of their organizations).
        """
        path = "{}/api/repos/ls".format(self.endpoint)
        params = {"organization": organization} if organization is not None else None
        r = requests.get(path, params=params, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()
        d = r.json()
        return [RepoObj(**x) for x in d]

    def create_repo(
        self,
        token: str,
        name: str,
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        exist_ok=False,
        lfsmultipartthresh: Optional[int] = None,
    ) -> str:
        """
        HuggingFace git-based system, used for models.

        Call HF API to create a whole repo.

        Params:
            private: Whether the model repo should be private (requires a paid huggingface.co account)

            exist_ok: Do not raise an error if repo already exists

            lfsmultipartthresh: Optional: internal param for testing purposes.
        """
        path = "{}/api/repos/create".format(self.endpoint)
        json = {"name": name, "organization": organization, "private": private}
        if lfsmultipartthresh is not None:
            json["lfsmultipartthresh"] = lfsmultipartthresh
        r = requests.post(
            path,
            headers={"authorization": "Bearer {}".format(token)},
            json=json,
        )
        if exist_ok and r.status_code == 409:
            return ""
        r.raise_for_status()
        d = r.json()
        return d["url"]

    def delete_repo(self, token: str, name: str, organization: Optional[str] = None):
        """
        HuggingFace git-based system, used for models.

        Call HF API to delete a whole repo.

        CAUTION(this is irreversible).
        """
        path = "{}/api/repos/delete".format(self.endpoint)
        r = requests.delete(
            path,
            headers={"authorization": "Bearer {}".format(token)},
            json={"name": name, "organization": organization},
        )
        r.raise_for_status()


class TqdmProgressFileReader:
    """
    Wrap an io.BufferedReader `f` (such as the output of `open(…, "rb")`) and override `f.read()` so as to display a
    tqdm progress bar.

    see github.com/huggingface/transformers/pull/2078#discussion_r354739608 for implementation details.
    """

    def __init__(self, f: io.BufferedReader):
        self.f = f
        self.total_size = os.fstat(f.fileno()).st_size
        self.pbar = tqdm(total=self.total_size, leave=False)
        self.read = f.read
        f.read = self._read

    def _read(self, n=-1):
        self.pbar.update(n)
        return self.read(n)

    def close(self):
        self.pbar.close()


class HfFolder:
    path_token = expanduser("~/.huggingface/token")

    @classmethod
    def save_token(cls, token):
        """
        Save token, creating folder as needed.
        """
        os.makedirs(os.path.dirname(cls.path_token), exist_ok=True)
        with open(cls.path_token, "w+") as f:
            f.write(token)

    @classmethod
    def get_token(cls):
        """
        Get token or None if not existent.
        """
        try:
            with open(cls.path_token, "r") as f:
                return f.read()
        except FileNotFoundError:
            pass

    @classmethod
    def delete_token(cls):
        """
        Delete token. Do not fail if token does not exist.
        """
        try:
            os.remove(cls.path_token)
        except FileNotFoundError:
            pass
