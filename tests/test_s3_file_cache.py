import boto3
import polars as pl
from moto import mock_aws

from experimental.run_stacking2 import S3FileCache


@mock_aws
def test_exists_returns_false_when_empty():
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-s3")
    cache = S3FileCache("s3://test-s3/results")
    assert not cache.exists("missing.parquet")


@mock_aws
def test_add_then_exists():
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-s3")
    cache = S3FileCache("s3://test-s3/results")

    df = pl.DataFrame({"a": [1, 2, 3]})
    cache.add(df, "test.parquet")
    assert cache.exists("test.parquet")


@mock_aws
def test_add_writes_readable_parquet():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-s3")
    cache = S3FileCache("s3://test-s3/results")

    df = pl.DataFrame({"x": [10, 20], "y": ["a", "b"]})
    cache.add(df, "roundtrip.parquet")

    obj = s3.get_object(Bucket="test-s3", Key="results/roundtrip.parquet")
    result = pl.read_parquet(obj["Body"].read())
    assert result.frame_equal(df)


@mock_aws
def test_exists_detects_preexisting_files():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-s3")
    s3.put_object(Bucket="test-s3", Key="results/old.parquet", Body=b"data")

    cache = S3FileCache("s3://test-s3/results")
    assert cache.exists("old.parquet")
