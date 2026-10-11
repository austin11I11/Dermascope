# Sample images

Used by the "Try a sample image" row in the app.

| File | True diagnosis | Source |
|---|---|---|
| `ISIC_0025314.jpg` | Dermatofibroma | HAM10000 |
| `ISIC_0027856.jpg` | Vascular lesion | HAM10000 |
| `ISIC_0027132.jpg` | Melanocytic nevus | HAM10000 |
| `ISIC_0031424.jpg` | Benign keratosis | HAM10000 |
| `ISIC_0027337.jpg` | Basal cell carcinoma | HAM10000 |
| `coffee_mug.jpg` | Not a skin lesion | Wikimedia Commons |

The five HAM10000 images are in the model's test split, so they were not used
for training, for validation, or in the out-of-distribution bank.

HAM10000: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a
large collection of multi-source dermatoscopic images of common pigmented skin
lesions. Sci. Data 5, 180161 (2018). Licensed CC BY-NC 4.0.

`coffee_mug.jpg`: "Coffee in a mug" by Kurt Kaiser, released under CC0 1.0
(public domain dedication),
https://commons.wikimedia.org/wiki/File:Coffee_in_a_mug.jpg. Resized to 600×450.
