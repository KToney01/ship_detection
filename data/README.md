# Data

This project uses the [ShipsNet satellite imagery dataset](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery).

ShipsNet was created from Planet satellite imagery made available through Planet's Open California dataset and is distributed under the CC BY-SA 4.0 license.

Dataset files are not included in this repository. Download ShipsNet separately and place the image files in a local directory, for example:

```text
~/data/shipsnet/
```

The dataset contains 4,000 80x80 RGB satellite images chips:

- 1,000 ship images
- 3,000 no-ship images

Images were extracted from Planet satellite imagery over the San Franciso Bay and San Pedro Bay areas.

## Local Structure

Raw downloaded data should be placed in:

data/raw/

The contents of 'data/raw/' are intentionally excluded from Git and must be downloaded separately.

## License

