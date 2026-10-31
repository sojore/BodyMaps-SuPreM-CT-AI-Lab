"""BodyMaps research viewer: reference masks vs real SuPreM predictions."""
from pathlib import Path
import io
import zipfile
import numpy as np
import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
ORGANS = ['liver', 'spleen', 'pancreas', 'kidney_left', 'kidney_right', 'aorta', 'stomach', 'gall_bladder', 'postcava']
WINDOWS = {'Soft tissue': (40, 400), 'Lung': (-600, 1500), 'Bone': (300, 1500)}

def read_nifti(blob):
    import tempfile
    with tempfile.TemporaryDirectory(prefix='bodymaps_nifti_') as folder:
        path = Path(folder) / 'volume.nii.gz'
        path.write_bytes(blob)
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError('Expected a 3D NIfTI volume.')
        data = np.asarray(image.dataobj)
        if not np.isfinite(data).all():
            raise ValueError('NIfTI contains non-finite values.')
        return data, image.affine.copy()

def zip_members(blob):
    result = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        entries = archive.infolist()
        if sum(x.file_size for x in entries) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError('Uncompressed ZIP exceeds the 1 GB safety limit.')
        for item in entries:
            if item.is_dir() or not item.filename.lower().endswith('.nii.gz'):
                continue
            if item.file_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError('Oversized NIfTI entry.')
            normalized = item.filename.replace('\\', '/')
            parts = Path(normalized).parts
            if '..' in parts or normalized.startswith('/'):
                raise ValueError('Unsafe archive path.')
            basename = Path(normalized).name
            if basename == 'ct.nii.gz':
                key = 'ct'
            elif basename in [f'{o}.nii.gz' for o in ORGANS]:
                key = basename[:-7]
            else:
                continue
            if key in result:
                raise ValueError(f'Duplicate NIfTI entry for {key}.')
            result[key] = archive.read(item)
    return result

@st.cache_data(show_spinner=False, max_entries=3)
def load_ct(blob, filename):
    if filename.lower().endswith('.zip'):
        items = zip_members(blob)
        if 'ct' not in items:
            raise ValueError('The original CT ZIP must contain ct.nii.gz.')
        ct, affine = read_nifti(items.pop('ct'))
        refs = {name: read_nifti(raw) for name, raw in items.items()}
        return ct, affine, refs
    ct, affine = read_nifti(blob)
    return ct, affine, {}

@st.cache_data(show_spinner=False, max_entries=3)
def load_predictions(blob):
    items = zip_members(blob)
    if 'ct' in items:
        raise ValueError('Upload the prediction ZIP, not the original CT ZIP, here.')
    if not items:
        raise ValueError('No supported SuPreM organ masks found in prediction ZIP.')
    return {name: read_nifti(raw) for name, raw in items.items()}

def validate_mask(name, mask_tuple, shape, affine):
    data, mask_affine = mask_tuple
    if data.shape != shape:
        raise ValueError(f'{name}: mask shape {data.shape} differs from CT {shape}.')
    if not np.allclose(mask_affine, affine, atol=1e-4):
        raise ValueError(f'{name}: mask affine differs from CT; resampling is required.')
    return data > 0

def segmentation_metrics(ref, pred):
    tp = int(np.count_nonzero(ref & pred))
    fp = int(np.count_nonzero(~ref & pred))
    fn = int(np.count_nonzero(ref & ~pred))
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    return {
        'Dice': (2 * tp / denom_dice) if denom_dice else 1.0,
        'IoU': (tp / denom_iou) if denom_iou else 1.0,
        'TP voxels': tp, 'FP voxels': fp, 'FN voxels': fn
    }

ORGAN_COLORS = {
    'liver': '#e76f51', 'spleen': '#9b5de5', 'pancreas': '#f4a261',
    'kidney_left': '#00b4d8', 'kidney_right': '#48cae4',
    'aorta': '#ef476f', 'stomach': '#ffd166',
    'gall_bladder': '#06d6a0', 'postcava': '#90be6d'
}

def get_plane(volume, plane, idx):
    axis = {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}[plane]
    return np.rot90(np.take(volume, idx, axis=axis))

def draw_plane(ax, ct, masks, plane, idx, center, width, alpha, title, spacing):
    from matplotlib.colors import to_rgb
    base = get_plane(ct, plane, idx)
    ax.imshow(base, cmap='gray', vmin=center-width/2, vmax=center+width/2, interpolation='nearest')
    for organ, mask in masks.items():
        sl = get_plane(mask, plane, idx)
        if not np.any(sl):
            continue
        rgba = np.zeros((*sl.shape, 4), dtype=np.float32)
        palette = {'True positive': '#2ecc71', 'False positive': '#ff4757', 'False negative': '#ffd166'}
        rgba[..., :3] = to_rgb(palette.get(organ, ORGAN_COLORS.get(organ, '#ffffff')))
        rgba[..., 3] = sl.astype(np.float32) * alpha
        ax.imshow(rgba, interpolation='nearest')
    remaining = [a for a in range(3) if a != {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}[plane]]
    ax.set_aspect(spacing[remaining[1]] / spacing[remaining[0]])
    ax.set_title(title, fontsize=11)
    ax.axis('off')

st.set_page_config(page_title='BodyMaps • CT AI Lab', page_icon='🫁', layout='wide')
st.title('BodyMaps | CT AI Lab')
st.caption('Research-only prototype • Real SuPreM predictions • Not for clinical decision-making')
st.info('Research-only GPU-hosted inference. Upload anonymized test CTs only; do not deploy this app publicly.')

import hashlib
from inference_backend import SuPremConfig, run_inference

workflow = st.radio('Workflow', ['Run SuPreM on uploaded CT (GPU host)', 'Inspect previously generated predictions'], horizontal=True)
ct_upload = st.file_uploader('1. Upload anonymized CT ZIP (optionally including reference masks) or CT NIfTI', type=['zip', 'gz'], key='ct')
pred_upload = None
if workflow == 'Inspect previously generated predictions':
    pred_upload = st.file_uploader('2. Upload SuPreM predictions ZIP', type=['zip'], key='pred')

if ct_upload is not None and workflow.startswith('Run SuPreM'):
    ct_blob = ct_upload.getvalue()
    upload_fingerprint = hashlib.sha256(ct_blob).hexdigest()
    if st.button('Run pretrained SuPreM inference', type='primary'):
        try:
            with st.spinner('Running real SuPreM inference on GPU server; large CT volumes may take time...'):
                config = SuPremConfig.from_environment()
                output_zip, generated_organs = run_inference(ct_blob, ct_upload.name, config)
            st.session_state['suprem_latest'] = (upload_fingerprint, output_zip)
            st.session_state['suprem_latest_name'] = f"{Path(ct_upload.name).stem}_SuPreM_predictions.zip"
            st.success(f'Inference completed: {len(generated_organs)} organ masks generated.')
        except Exception as exc:
            st.session_state.pop('suprem_latest', None)
            st.error(f'Inference unsuccessful: {exc}')

if workflow.startswith('Run SuPreM'):
    prior_download = st.session_state.get('suprem_latest')
    if prior_download:
        st.download_button('Download generated SuPreM masks', prior_download[1],
                           file_name=st.session_state.get('suprem_latest_name', 'SuPreM_predictions.zip'),
                           mime='application/zip')

if ct_upload is not None:
    try:
        if ct_upload.size > MAX_UPLOAD_BYTES or (pred_upload and pred_upload.size > MAX_UPLOAD_BYTES):
            raise ValueError('Upload exceeds the 300 MB limit.')
        with st.spinner('Reading CT and masks...'):
            ct, affine, refs_raw = load_ct(ct_upload.getvalue(), ct_upload.name)
            pred_blob = pred_upload.getvalue() if pred_upload else None
            if workflow.startswith('Run SuPreM'):
                prior = st.session_state.get('suprem_latest')
                pred_blob = prior[1] if prior and prior[0] == hashlib.sha256(ct_upload.getvalue()).hexdigest() else None
            preds_raw = load_predictions(pred_blob) if pred_blob else {}
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        refs = {name: validate_mask('Reference '+name, value, ct.shape, affine) for name, value in refs_raw.items()}
        preds = {name: validate_mask('Prediction '+name, value, ct.shape, affine) for name, value in preds_raw.items()}
        st.success(f'CT loaded: {ct.shape} • {len(refs)} reference masks • {len(preds)} AI-predicted masks')

        available = [o for o in ORGANS if o in refs or o in preds]
        controls, viewer = st.columns([1, 3])
        with controls:
            st.subheader('Viewer controls')
            view_mode = st.radio('Display', ['Three-plane viewer', 'Single-plane viewer'], horizontal=False)
            mask_source = st.radio('Overlay source', ['AI predictions', 'Reference masks', 'Error analysis', 'CT only'])
            error_organ = None
            if mask_source == 'Error analysis':
                eligible = [o for o in ORGANS if o in refs and o in preds]
                if eligible:
                    error_organ = st.selectbox('Organ for error analysis', eligible)
                else:
                    st.warning('Both reference and prediction masks are required.')
            window = st.selectbox('CT window', list(WINDOWS), index=0)
            chosen = st.multiselect('Organs (multiple allowed)', available,
                                    default=[o for o in ['liver','spleen','kidney_left','kidney_right'] if o in available])
            alpha = st.slider('Overlay opacity', 0.1, 0.9, 0.40)
            st.caption('Each organ has a unique color. Select CT only to inspect the unannotated image.')
            x = st.slider('Sagittal index (axis 0)', 0, ct.shape[0]-1, ct.shape[0]//2)
            y = st.slider('Coronal index (axis 1)', 0, ct.shape[1]-1, ct.shape[1]//2)
            z = st.slider('Axial index (axis 2)', 0, ct.shape[2]-1, ct.shape[2]//2)
            if view_mode == 'Single-plane viewer':
                selected_plane = st.selectbox('Plane', ['Axial', 'Coronal', 'Sagittal'])
            source = preds if mask_source == 'AI predictions' else refs if mask_source == 'Reference masks' else {}
            if mask_source == 'Error analysis' and error_organ:
                reference, prediction = refs[error_organ], preds[error_organ]
                overlays = {
                    'True positive': reference & prediction,
                    'False positive': ~reference & prediction,
                    'False negative': reference & ~prediction
                }
            else:
                overlays = {o: source[o] for o in chosen if o in source}
            if mask_source not in ('CT only', 'Error analysis') and not source:
                st.warning('No masks available for the selected overlay source.')
            if mask_source not in ('CT only', 'Error analysis') and chosen and not overlays:
                st.warning('None of the selected organs is available in this mask source.')

        with viewer:
            center, width = WINDOWS[window]
            indices = {'Axial': z, 'Coronal': y, 'Sagittal': x}
            if view_mode == 'Three-plane viewer':
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                for ax, plane in zip(axes, ['Axial', 'Coronal', 'Sagittal']):
                    draw_plane(ax, ct, overlays, plane, indices[plane], center, width, alpha,
                               f'{plane} • slice {indices[plane]}', spacing)
            else:
                fig, ax = plt.subplots(figsize=(8, 7))
                draw_plane(ax, ct, overlays, selected_plane, indices[selected_plane],
                           center, width, alpha, f'{selected_plane} • slice {indices[selected_plane]}', spacing)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            if overlays:
                from matplotlib.patches import Patch
                palette = {'True positive': '#2ecc71', 'False positive': '#ff4757', 'False negative': '#ffd166'}
                legend_items = [Patch(facecolor=palette.get(o, ORGAN_COLORS.get(o, '#ffffff')),
                                      label=o.replace('_', ' ')) for o in overlays]
                if legend_items:
                    legend_fig, legend_ax = plt.subplots(figsize=(10, 0.9))
                    legend_ax.axis('off')
                    legend_ax.legend(handles=legend_items, loc='center', ncol=min(5,len(legend_items)),
                                     frameon=False, fontsize=9)
                    st.pyplot(legend_fig, use_container_width=True)
                    plt.close(legend_fig)
            if mask_source == 'Error analysis' and error_organ:
                stats = segmentation_metrics(refs[error_organ], preds[error_organ])
                st.caption('Error overlay: green = agreement on organ voxels (TP), red = extra predicted voxels (FP), yellow = reference voxels missed by AI (FN). Background agreement is not highlighted.')
                a, b, c = st.columns(3)
                a.metric('TP voxels', f"{stats['TP voxels']:,}")
                b.metric('FP voxels', f"{stats['FP voxels']:,}")
                c.metric('FN voxels', f"{stats['FN voxels']:,}")
            st.caption('Physical voxel spacing is used to preserve anatomical proportions. Plane labels refer to NIfTI voxel axes (0, 1, 2). Anatomical orientation depends on the input affine; this is a voxel-axis viewer, not a clinically oriented radiology display.')

        comparable = [o for o in ORGANS if o in refs and o in preds]
        if comparable:
            st.subheader('Segmentation accuracy on this CT')
            rows = []
            for o in comparable:
                metrics = segmentation_metrics(refs[o], preds[o])
                rows.append({
                    'Organ': o, 'Dice': round(metrics['Dice'], 4),
                    'IoU': round(metrics['IoU'], 4),
                    'TP voxels': metrics['TP voxels'],
                    'FP voxels': metrics['FP voxels'],
                    'FN voxels': metrics['FN voxels'],
                    'Reference voxels': int(refs[o].sum()),
                    'Predicted voxels': int(preds[o].sum())
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric('Mean Dice (available organs)', f"{np.mean([r['Dice'] for r in rows]):.4f}")
            m2.metric('Mean IoU (available organs)', f"{np.mean([r['IoU'] for r in rows]):.4f}")
            import pandas as pd
            st.download_button('Download evaluation CSV',
                               pd.DataFrame(rows).to_csv(index=False).encode('utf-8'),
                               file_name='bodymaps_segmentation_metrics.csv', mime='text/csv')
            st.caption('Single-scan demonstration only. Not a held-out benchmark or clinical validation.')
        elif preds:
            st.info('Predictions loaded. Upload the original ZIP with reference masks to calculate Dice.')
    except Exception as exc:
        st.error(f'Could not process upload: {exc}')
else:
    st.caption('Start by uploading one anonymized research CT or a CT-and-mask ZIP.')
