"""Unit tests for the reserved identity contract."""
from __future__ import annotations

import pytest

from . import ids


def test_validate_identity_happy():
    ids.validate_identity(alias="llama3-8b-parity", chip="gpu",
                          region="ap-northeast-1", workload_id="prefill-bs1")


@pytest.mark.parametrize("bad", ["", "has space", "has/slash", "quote'd", "a" * 200, "-leadingdash"])
def test_validate_identity_rejects_bad_alias(bad):
    with pytest.raises(ValueError):
        ids.validate_identity(alias=bad, chip="gpu", region="ap-northeast-1", workload_id="w")


def test_validate_identity_rejects_bad_chip():
    with pytest.raises(ValueError):
        ids.validate_identity(alias="a", chip="tpu", region="ap-northeast-1", workload_id="w")


@pytest.mark.parametrize("bad", ["", "US-EAST-1", "region with space"])
def test_validate_identity_rejects_bad_region(bad):
    with pytest.raises(ValueError):
        ids.validate_identity(alias="a", chip="neuron", region=bad, workload_id="w")


def test_reserved_tags_shape():
    t = ids.reserved_tags(alias="a", chip="gpu", region="ap-northeast-1",
                          workload_id="w", artifacts_uri="s3://b/a/r/")
    assert set(t) == set(ids.RESERVED_TAGS)
    assert t[ids.SCHEMA_VERSION_TAG] == ids.SCHEMA_VERSION
    assert t[ids.CHIP_TAG] == "gpu"


def test_is_reserved():
    assert ids.is_reserved("chip")
    assert ids.is_reserved("artifacts_uri")
    assert not ids.is_reserved("cosine")
