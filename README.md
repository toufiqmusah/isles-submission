# ISLES 2026: Template Docker for Algorithm Submission

Welcome to the official Docker submission template for **ISLES 2026**. Follow the instructions below to configure, build, and test your algorithm before submission.

---

## Instructions

### 1. Configure Your Algorithm
Modify the following components to integrate your model:
* **`inference.py`**: Update this file with your custom algorithm and prediction logic.
* **`requirements.txt`**: Add any required Python packages and dependencies.
* **`model/`**: Place your trained model weights/checkpoints inside this directory.

---

### 2. Build the Docker Image
Build your Docker image locally by executing the build script:

```bash
./do_build.sh
```

---

### 3. Test Your Container

Before submitting, verify that your container runs correctly locally:

1. **Input Data Placement:** Place the test MRI scan (`.nii.gz`) inside the input directory:
   ```text
   test/input/interf0/images/t1-brain-mri/
   ```
2. **Metadata Setup (If applicable):** If your algorithm relies on clinical metadata, update:
   ```text
   test/input/interf0/stroke-metadata.json
   ```
3. **Execute Test:** Run the container test script:
   ```bash
   ./do_test_run.sh
   ```
4. **Verify Output:** If the execution completes successfully, check your predictions generated under:
   ```text
   test/output/
   ```
   Ensure the output is as you expect.

---

### 4. Guidelines for Using the Metadata [optional]

* **JSON Format:** Grand Challenge requires metadata files in `.json` format. Convert your train set `.csv` files using the provided template:
  ```bash
  metadata-csv_to_json.py
  ```
* **Column Name Changes:** Note that the `SITE` column in the original `.csv` files is mapped to `CENTER` in the `.json` files. **You must reference `'CENTER'` in `inference.py`**, as this reflects how Grand Challenge formats the metadata payload.

---

## Helpful Resources

* For additional technical details, consult the [Grand Challenge Algorithm Creation Documentation](https://grand-challenge.org/documentation/create-your-own-algorithm/).
