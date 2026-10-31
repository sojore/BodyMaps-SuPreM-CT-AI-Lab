from dataclasses import dataclass
from pathlib import Path
import io, os, re, subprocess, tempfile, zipfile, hashlib

ORGANS = ['liver','spleen','pancreas','kidney_left','kidney_right',
          'aorta','stomach','gall_bladder','postcava']
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024

@dataclass(frozen=True)
class SuPremConfig:
    repo: Path
    checkpoint: Path
    python: Path
    report_dir: Path

    @classmethod
    def from_environment(cls):
        repo = Path(os.environ['SUPREM_REPO']).resolve()
        checkpoint = Path(os.environ['SUPREM_CHECKPOINT']).resolve()
        python = Path(os.environ['SUPREM_PYTHON'])
        report_dir = Path(os.environ.get('BODYMAPS_REPORT_DIR','/content/bodymaps_reports')).resolve()
        for p, label in [(repo,'SuPreM repo'),(checkpoint,'checkpoint'),(python,'Python interpreter')]:
            if not p.exists():
                raise RuntimeError(f'{label} not found: {p}')
        verified = repo/'direct_inference'/'inference_bodymaps_verified.py'
        if not verified.is_file():
            raise RuntimeError('Verified inference script is missing. Run the checkpoint-audit cells first.')
        report_dir.mkdir(parents=True, exist_ok=True)
        return cls(repo, checkpoint, python, report_dir)

def _safe_case_id(filename, blob):
    name = filename[:-4] if filename.lower().endswith('.zip') else filename
    if name.lower().endswith('.nii.gz'):
        name = name[:-7]
    name = re.sub(r'[^A-Za-z0-9_.-]+','_',name).strip('._-')
    if not name or name.lower() in {'ct','scan','upload'}:
        name = 'browser_' + hashlib.sha256(blob).hexdigest()[:10]
    return name[:80]

def _extract_ct(blob, filename, destination):
    if len(blob) > MAX_UPLOAD_BYTES:
        raise ValueError('Upload exceeds the 300 MiB safety limit.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if filename.lower().endswith('.nii.gz'):
        if not blob.startswith(b'\\x1f\\x8b'):
            raise ValueError('The uploaded .nii.gz is not gzip-compressed.')
        destination.write_bytes(blob)
        return
    if not filename.lower().endswith('.zip'):
        raise ValueError('Upload must be a .zip or .nii.gz CT.')
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = zf.infolist()
        total = sum(x.file_size for x in members)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError('ZIP expands beyond the 1 GiB safety limit.')
        matches = []
        for item in members:
            if item.is_dir():
                continue
            normalized = item.filename.replace('\\\\','/')
            parts = Path(normalized).parts
            if normalized.startswith('/') or '..' in parts:
                raise ValueError('Unsafe archive path.')
            if Path(normalized).name.lower() == 'ct.nii.gz':
                matches.append(item)
        if len(matches) != 1:
            raise ValueError(f'ZIP must contain exactly one ct.nii.gz; found {len(matches)}.')
        if matches[0].file_size > MAX_UPLOAD_BYTES:
            raise ValueError('CT NIfTI exceeds the upload safety limit.')
        destination.write_bytes(zf.read(matches[0]))

def run_inference(blob, filename, config):
    case_id = _safe_case_id(filename, blob)
    with tempfile.TemporaryDirectory(prefix='bodymaps_web_') as tmp:
        root = Path(tmp)
        data_root = root/'input'
        save_root = root/'output'
        case_dir = data_root/case_id
        case_dir.mkdir(parents=True)
        _extract_ct(blob, filename, case_dir/'ct.nii.gz')
        script = config.repo/'direct_inference'/'inference_bodymaps_verified.py'
        cmd = [
            str(config.python), '-W', 'ignore', str(script),
            '--save_dir', str(save_root),
            '--checkpoint', str(config.checkpoint),
            '--data_root_path', str(data_root),
            '--backbone', 'unet', '--store_result', '--suprem',
            '--batch_size', '1', '--num_workers', '0',
            '--roi_x', '96', '--roi_y', '96', '--roi_z', '96',
            '--overlap', '0.25'
        ]
        env = os.environ.copy()
        env.update(MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
        result = subprocess.run(
            cmd, cwd=str(config.repo/'direct_inference'), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15*60
        )
        log_path = config.report_dir/f'web_inference_{case_id}.log'
        log_path.write_text(result.stdout or '', encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError(
                'SuPreM inference failed. The private Colab log contains details; '
                'no internal server paths are exposed in the web UI.'
            )
        seg_dir = save_root/case_id/'segmentations'
        missing = [o for o in ORGANS if not (seg_dir/f'{o}.nii.gz').is_file()]
        if missing:
            raise RuntimeError('Inference finished but required masks are missing: ' + ', '.join(missing))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for organ in ORGANS:
                zf.write(seg_dir/f'{organ}.nii.gz', arcname=f'{organ}.nii.gz')
        return buffer.getvalue(), ORGANS.copy()
