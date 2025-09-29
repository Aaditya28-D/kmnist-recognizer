# KMNIST Recognizer 📝  

This project recognizes **handwritten Japanese Kana characters** using AI (Deep Learning).  
You can upload, draw, or sample characters and the model will predict them.  

👉 **Try it online (no setup needed):**  
🔗 [KMNIST Recognizer on Hugging Face](https://huggingface.co/spaces/Aaditya28D/kmnist-inference)  

---

## 🚀 How to Run Locally  

If you want to run this project on your own computer:  

### 1. Get the project  
Download this project to your computer (from GitHub):  

```bash
git clone https://github.com/Aaditya28-D/kmnist-recognizer.git
cd kmnist-recognizer
```

*(If you don’t have Git, you can also click **Download ZIP** on GitHub and unzip it.)*

---

### 2. Install requirements  
Install all needed Python libraries:  

```bash
pip install -r requirements.txt
```

---

### 3. Download model & data files  
Run this script once to set up **test data** and download the **latest trained models** from Google Drive:  

```bash
python3 setup_assets.py
```

This will:  
- Download `kmnist-train` and `kmnist-test` data automatically.  
- Fetch the **latest model weights** listed in `models_index.json`.  
- Store them in the `models/` folder.  

---

### (Optional) Upload new models  
If you train new models and want to save them to Drive for sharing/reproducibility:  

```bash
python upload_models.py --folder-id <your-drive-folder-id>
```

This updates `models_index.json` so others can download the new model with `setup_assets.py`.  

---

### 4. Start the app  
Finally, run:  

```bash
python app.py
```

You will see something like:  

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxx.gradio.live
```

- Open the **local URL** in your browser if running on your computer.  
- Share the **public URL** with others so they can try it too.  

---

## 📝 Notes  
- No training data is included — only test data and the trained model.  
- Sometimes predictions may not be 100% accurate. This is normal for AI models.  
- Click the ⤢ button on any image to zoom in.  
- If you don’t want to install anything, just use the Hugging Face link above.  
