# 4AL3-Assignment2

## How to Run  
Run the script from terminal with **three arguments**:  
```bash
python assignment22.py "<path_to_2010_data>" "<path_to_2020_data>" "<output_folder>"
```
Example:
```bash
python assignment22.py "C:\Users\grace\4AL3-Assignment2\data-2010-15" "C:\Users\grace\4AL3-Assignment2\data-2020-24" "C:\Users\grace\4AL3-Assignment2\output"
```

## Dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## Output:
Results (plots, CSVs, confusion matrices) will be saved automatically under:
```bash
output/
 ├─ reports_2010/
 └─ reports_2020/
```

## Use of Generative AI
I used ChatGPT (OpenAI GPT-5, cloud-based model) to:
- Clarify assignment instructions and deliverables
- Refactor and debug the Python implementation (assignment22.py)
- Suggest improvements for file handling and figure generation
- Help draft this README file

I did not use ChatGPT to generate experimental results or perform analysis — all experiments and report writing were completed independently on my local machine.

## Carbon Footprint Estimate
Following the course policy:
- Model: ChatGPT (GPT-5)
- Hardware type: Cloud GPU/TPU (provider: OpenAI)
- Provider: OpenAI
- Region of compute: Hamilton region
- Time used: ~3 days × 8 hours / day ≈ 24 hours total
- Approximate estimate: ~400 queries × 4.32 g CO₂ ≈ 1.73 kg CO₂