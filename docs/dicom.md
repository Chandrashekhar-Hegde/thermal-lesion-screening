# DICOM validation

DMR-IR is distributed as ordinary image files, so the training path uses
`manifest_from_folders()`. The DICOM path is checked separately against a fixed public
breast-MR series from the NCI Imaging Data Commons I-SPY1 collection.

## Public series

- Collection: [I-SPY1](https://www.cancerimagingarchive.net/collection/ispy1/)
- License: CC BY 3.0
- Source DOI: [10.7937/K9/TCIA.2016.HdHpgJLK](https://doi.org/10.7937/K9/TCIA.2016.HdHpgJLK)
- Series Instance UID:
  `1.3.6.1.4.1.14519.5.2.1.7695.1700.153974929648969296590126728101`
- Size: three MR instances, approximately 0.1 MB

Download it without an account using the
[official IDC client](https://github.com/ImagingDataCommons/idc-index):

```bash
uvx --from idc-index idc download \
  '1.3.6.1.4.1.14519.5.2.1.7695.1700.153974929648969296590126728101' \
  --download-dir data/public-dicom
```

Validate metadata and decoded pixels:

```bash
python -m scripts.validate_dicom_ingest data/public-dicom \
  --source 'NCI IDC I-SPY1, DOI 10.7937/K9/TCIA.2016.HdHpgJLK' \
  --series-uid '1.3.6.1.4.1.14519.5.2.1.7695.1700.153974929648969296590126728101' \
  --output docs/validation/ispy1_dicom_ingest.json
```

The committed summary was generated on 31 August 2026. It contains no local paths or
patient names.

## Orthanc round trip

This runbook follows the [official Orthanc Docker guide](https://orthanc.uclouvain.be/book/users/docker.html)
and [REST API documentation](https://orthanc.uclouvain.be/book/users/rest.html). Bind the
server to loopback; do not expose the example credentials.

```bash
docker run --rm --name thermal-lesion-orthanc \
  -p 127.0.0.1:8042:8042 \
  -e ORTHANC__AUTHENTICATION_ENABLED=true \
  -e ORTHANC__REGISTERED_USERS='{"reviewer":"local-only-password"}' \
  jodogne/orthanc:1.13.0
```

In another shell, upload the downloaded series:

```bash
find data/public-dicom -name '*.dcm' -print0 | while IFS= read -r -d '' file; do
  curl --fail --silent --show-error -u reviewer:local-only-password \
    -X POST -H 'Expect:' --data-binary "@$file" \
    http://127.0.0.1:8042/instances
done
```

List the study, download its archive, and scan the retrieved files:

```bash
study_id=$(curl --fail --silent -u reviewer:local-only-password \
  http://127.0.0.1:8042/studies | python -c 'import json,sys; print(json.load(sys.stdin)[0])')
curl --fail --silent -u reviewer:local-only-password \
  "http://127.0.0.1:8042/studies/$study_id/archive" \
  --output artifacts/orthanc-study.zip
unzip -q artifacts/orthanc-study.zip -d artifacts/orthanc-study
python -m scripts.validate_dicom_ingest artifacts/orthanc-study \
  --source 'Orthanc round trip of NCI IDC I-SPY1 series'
```

The public-series ingest has been executed and its summary is committed. The Orthanc
round trip remains a documented local check because Docker or Orthanc was not installed
on the validation host. Do not describe it as completed until its output is recorded.
