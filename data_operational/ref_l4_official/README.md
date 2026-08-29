---
license: cc-by-nc-4.0
language:
  - en
tags:
  - referring expression comprehension
  - large multimodal model
size_categories:
  - 10K<n<100K
configs:
  - config_name: ref_l4
    data_files:
      - split: val
        path: ref-l4-val.parquet
      - split: test
        path: ref-l4-test.parquet
---

# Ref-L4

## Introduction
Referring expression comprehension (REC) involves localizing a target instance based on a textual description. Recent advancements in REC have been driven by large multimodal models (LMMs) like CogVLM, which achieved 92.44% accuracy on RefCOCO. However, this study questions whether existing benchmarks such as RefCOCO, RefCOCO+, and RefCOCOg, capture LMMs' comprehensive capabilities. These benchmarks vary in complexity, but our manual assessment reveals high labeling error rates: 14% in RefCOCO, 24% in RefCOCO+, and 5% in RefCOCOg, undermining the authenticity of evaluations. We address this by excluding problematic instances and reevaluating several LMMs capable of handling the REC task, showing significant accuracy improvements, thus highlighting the impact of benchmark noise. In response, we introduce Ref-L4, a comprehensive REC benchmark, specifically designed to evaluate modern REC models. Ref-L4 is distinguished by four key features:

1. A substantial sample size with **45,341 annotations**
2. A diverse range of object categories with **365 distinct types** and varying instance scales from 30 to 3,767
3. Lengthy referring expressions averaging **24.2 words**
4. An extensive vocabulary comprising **22,813 unique words**

<img src="figs/examples.png" align="center" width="800" />

## Dataloader & Evaluation Protocol
- The **dataloader** and **evaluation** APIs are available at the [Ref-L4 GitHub repository](https://github.com/JierunChen/Ref-L4). Additionally, several demonstrations for evaluating LMMs can be found in the repository.

## File Tree

Our Ref-L4 dataset is organized as follows, including the images, val, and test splits. We also provide reviewed annotations of RefCOCO, RefCOCO+, and RefCOCOg benchmarks.

```bash
Ref-L4
├── images.tar.gz
├── README.md
├── refcocos_annotation_reviewed
│   ├── refcocog_test_reviewed.json
│   ├── refcocog_val_reviewed.json
│   ├── refcoco+_testA_reviewed.json
│   ├── refcoco_testA_reviewed.json
│   ├── refcoco+_testB_reviewed.json
│   ├── refcoco_testB_reviewed.json
│   ├── refcoco+_val_reviewed.json
│   └── refcoco_val_reviewed.json
├── ref-l4-test.parquet
└── ref-l4-val.parquet
```
## Reviewed RefCOCO (+/g) Annotations

### Error Rates and Annotation Statistics
The following table summarizes the labeling error rates and the number of annotations for the RefCOCO, RefCOCO+, and RefCOCOg benchmarks:

| Benchmark | Annotations | Errors | Error Rate |
|:----------|:-----------:|:------:|:----------:|
| RefCOCO   | 21,586      | 3,054  | 14%        |
| RefCOCO+  | 21,373      | 5,201  | 24%        |
| RefCOCOg  | 14,498      | 675    | 5%         |

### Access to Reviewed Annotations
We provide the reviewed annotations of RefCOCO (+/g) under the directory `./refcocos_annotation_reviewed/`.

For each instance in the `'annotation'` field of the `refcoco(+/g)_[split]_reviewed.json` files, we denote erroneous entries with `["caption_quality"]=0`.

### Sample of labeling errors
In the REC task, a referring expression should uniquely describe an instance, which is represented by an accurate bounding box. We have identified and visualized three common types of labeling errors in the RefCOCO, RefCOCO+, and RefCOCOg benchmarks: (a) non-unique referring expressions, which refer to multiple instances within the same image; (b) inaccurate bounding boxes; and (c) misalignment between target instances and their referring expressions, where the referring expressions are either ambiguous or do not refer to any instance in the image.

<img src="figs/error_samples.png" align="center" width="800" />


## Annotation Format
The `ref-l4-val(test).parquet` file is a list of dictionaries, each representing an annotation for a particular image. Here is an example of one annotation item:
```json
{
  "id":1,
  "caption":"Within the central picture frame of the three, an antique camera is present.",
  "bbox":[580.6163330048,179.4965209869,93.59924316159993,112.1013793848],
  "bbox_area":10492.60426778866,
  "bbox_id":"o365_527361",
  "ori_category_id":"o365_64",
  "image_id":"o365_922765",
  "height":741,"width":1024,
  "file_name":"objects365_v2_00922765.jpg",
  "is_rewrite":true,
  "split":"val"
}
```
### Annotation Fields
- `id`: Unique identifier for the annotation.
- `caption`: A textual description or caption for the annotated object.
- `bbox`: Bounding box coordinates `[x, y, w, h]` of the annotated object.
- `bbox_area`: The area of the bounding box.
- `bbox_id`: Unique identifier for the box.
- `ori_category_id`: Original category identifier.
- `image_id`: Unique identifier for the image.
- `height`: Height of the image.
- `width`: Width of the image.
- `file_name`: The filename of the image.
- `is_rewrite`: Indicator if the caption is a rewritten version, `false` for raw caption and `true` for rewritten.
- `split`: Benchmark split ('val' or 'test').

## License
The Ref-L4 dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license](https://creativecommons.org/licenses/by-nc/4.0/). Please note that the images in the Ref-L4 dataset are derived from the following datasets, each with their respective licenses:
- **RefCOCO**: Licensed under the [Apache-2.0 license](http://www.apache.org/licenses/).
- **RefCOCO+**: Licensed under the [Apache-2.0 license](http://www.apache.org/licenses/).
- **RefCOCOg**: Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).
- **COCO 2014**: Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).
- **Objects365**: Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](http://creativecommons.org/licenses/by/4.0/).

By using the Ref-L4 dataset, you agree to comply with the licensing terms of these source datasets.
