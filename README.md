
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install requirements
pip install torch transformers datasets pandas

# 4. Run training (optional, pre-trained model used by default)
python model.py

# 5. Run inference
python inference.py
```