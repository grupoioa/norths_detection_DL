from enum import Enum


class FileNames(Enum):
    # Original names in ITK
    PREPROC_IMG_ORIGINAL = 'img_orig.nrrd'
    PREPROC_CTR_ORIGINAL = 'ctr_orig.nrrd'
    # Resampled names in ITK
    PREPROC_IMG_RESAMPLED = 'img_res.nrrd'
    PREPROC_CTR_RESAMPLED = 'ctr_res.nrrd'
    # Cropped names in ITK
    PREPROC_IMG_CROPPED = 'img_crop.nrrd'

    PREPROC_CTR_CROPPED_ALL = 'ctr_crop_both.nrrd'
    PREPROC_CTR_CROPPED_KIDNEY = 'ctr_crop_kidney.nrrd'
    PREPROC_CTR_CROPPED_TUMOR = 'ctr_crop_tumor.nrrd'
