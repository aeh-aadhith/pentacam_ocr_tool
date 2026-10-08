# Pentacam OCR Tool

This tool reads Pentacam printout images (JPG files) and pulls out the
numbers on them, such as K1, K2, KMax, Pachy and the indices. It saves the
results as a spreadsheet (CSV) and a JSON file.

The text reading (OCR) is done by the DeepSeek-OCR model, running on your
own computer. Patient images are never sent anywhere.

## Supported reports

The tool works with these three Pentacam reports. It detects the type of
each image automatically.

- Refractive
- Belin/Ambrosio Enhanced Ectasia
- 4 Maps Refractive

The images should be the normal Pentacam JPG exports, 1200 x 838 pixels in
size.

## How it works

1. **Finds the report type.** The tool reads the title at the top of each
   image to tell whether it is a Refractive, 4 Maps Refractive or
   Belin/Ambrosio report. You don't need to name or sort the images; you
   can mix all three types in one folder.
2. **Cuts the image into boxes.** Each report type has a known layout, so
   the tool crops out the boxes that hold the numbers and leaves out the
   colour maps.
3. **Reads each box separately.** OCR runs on one box at a time, which is
   more reliable than reading the whole page at once.
4. **Collects the results.** The values from all the boxes are combined
   into one entry per image and saved as JSON and CSV.

## What you need

- A Linux computer
- Python 3.8 or newer
- About 5 GB of free disk space
- An internet connection for the first run only

## Setup (one time)

1. Install the system tools:

   ```
   sudo apt install git cmake build-essential
   ```

2. Install the Python package:

   ```
   pip install -r requirements.txt
   ```

3. Prepare the OCR model:

   ```
   python3 pentacam_ocr.py --setup-only
   ```

   This downloads the model (about 4 GB) and builds the OCR program. It can
   take some time. If the download stops, run the same command again and it
   will continue from where it stopped.

   You can skip this step. The tool will do it automatically the first
   time you use it.

## How to use

Put the Pentacam JPG images in one folder, then run:

```
python3 pentacam_ocr.py "/path/to/your/image/folder"
```

Each image takes a few minutes. The tool shows its progress as it goes.

## Results

The results are saved inside the same image folder:

- `ocr_results.csv` - one row per image. Open it in Excel or any
  spreadsheet program.
- `ocr_results.json` - the same results in JSON format.

## Sample images

The `Samples` folder has one example of each report type, with the
results the tool produced for them:

- `Anonymous_OD_Refractive.JPG`
- `Anonymous_OD_4 Maps Refractive.JPG`
- `Anonymous_OD_BelinAmbrosio.JPG`
- `ocr_results.json` and `ocr_results.csv` - the results for these three
  images

Use them to see what the output looks like, or to check that the tool works
on your computer:

```
python3 pentacam_ocr.py Samples
```

## Good to know

- **Axis values:** The Belin/Ambrosio report shows the flat axis. The
  Refractive and 4 Maps reports show the steep axis. So they differ by
  90 degrees. This is normal.
- **Empty boxes** on the printout (for example Lens Th.) are left out of
  the results.
- **Privacy:** While reading the images, the tool turns off network access
  for itself, so patient data cannot leave the computer.

## Common problems

**"Refusing to run: cannot guarantee network isolation"**

This can happen on Ubuntu 24.04 and newer. The safest fix is to ask your
system administrator to run:

```
sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0
```

Or disconnect from the internet and run the tool with the extra option
`--no-network-isolation`.

**"Cannot build the OCR runtime: missing ..."**

Some system tools are not installed. Run step 1 of the setup again.

**"Pillow is required"**

Run step 2 of the setup again.

## Options

| Option | What it does |
|---|---|
| `--setup-only` | Only download and build the OCR model, then stop |
| `--out NAME` | Save the results with a different file name |
| `--pattern "*.jpg"` | Choose which image files to read (default is `*.JPG`) |
| `--no-network-isolation` | Run even if network access cannot be turned off |
