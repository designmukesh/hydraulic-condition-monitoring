"""Download UCI dataset 447 (~73.1 MB compressed) and record content hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip'
ROOT = Path(__file__).resolve().parents[1]


def record_manifest(archive):
    files = sorted((ROOT / 'data').glob('*.txt'))
    manifest = {'url': URL, 'doi': '10.24432/C5CW21', 'archive_bytes': archive.stat().st_size,
                'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    (ROOT / 'reports' / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    archive = ROOT / 'data' / 'hydraulic.zip'
    print('Downloading UCI dataset (~73.1 MB compressed)', flush=True)
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            destination = (archive.parent / member.filename).resolve()
            if not destination.is_relative_to(archive.parent.resolve()):
                raise ValueError('Unsafe archive member')
        z.extractall(archive.parent)
    record_manifest(archive)
    archive.unlink()
    print('Download complete; content hashes recorded in reports/data_manifest.json')


if __name__ == '__main__':
    main()
