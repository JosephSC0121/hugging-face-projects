
```bash
# 1. Create virtual enviroment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install requirements
pip install torch transformers datasets pandas

# 3. Run training (optional, pre-trained model used by default)
python model.py

# 1. Run inference
python inference.py
```