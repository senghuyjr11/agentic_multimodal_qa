"""
Verification Script: Check preprocessing is correct before retraining
Run this BEFORE starting training!
"""
import torch

def verify_preprocessed_data(data_path):
    print(f"Loading: {data_path}")
    data = torch.load(data_path, weights_only=False)
    print(f"Samples: {len(data)}\n")
    
    issues = []
    
    for idx in range(min(5, len(data))):
        sample = data[idx]
        labels = sample["labels"]
        input_ids = sample["input_ids"]
        
        print(f"{'='*60}")
        print(f"Sample {idx}")
        print(f"{'='*60}")
        
        # Check 1: EOS token in labels (NOT masked)
        eos_token_id = 151645  # Qwen2 EOS token
        last_label = labels[-1].item()
        
        if last_label == -100:
            print(f"❌ ISSUE: EOS token is MASKED (labels[-1] = -100)")
            print(f"   Model won't learn to stop generating!")
            issues.append(f"Sample {idx}: EOS masked")
        elif last_label == eos_token_id:
            print(f"✅ EOS token NOT masked (labels[-1] = {last_label})")
        else:
            print(f"⚠️  Last label is {last_label}, expected EOS={eos_token_id}")
        
        # Check 2: Answer tokens exist
        answer_mask = labels != -100
        num_answer_tokens = answer_mask.sum().item()
        print(f"   Answer tokens (trainable): {num_answer_tokens}")
        
        if num_answer_tokens == 0:
            print(f"❌ ISSUE: No answer tokens!")
            issues.append(f"Sample {idx}: No answer tokens")
        elif num_answer_tokens == 1:
            print(f"⚠️  Only 1 answer token (just EOS?)")
        
        # Check 3: Decode answer portion
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            
            answer_ids = labels[answer_mask]
            decoded = tokenizer.decode(answer_ids, skip_special_tokens=True)
            print(f"   Decoded answer: '{decoded}'")
            
            # Check for contamination
            if "assistant" in decoded.lower():
                print(f"❌ ISSUE: 'assistant' in answer!")
                issues.append(f"Sample {idx}: assistant contamination")
            if "<|" in decoded:
                print(f"❌ ISSUE: Special tokens in answer!")
                issues.append(f"Sample {idx}: special token contamination")
                
        except Exception as e:
            print(f"   (Could not decode: {e})")
        
        # Check 4: Image tensor shape
        pv = sample["pixel_values"]
        igt = sample["image_grid_thw"]
        print(f"   pixel_values shape: {pv.shape}")
        print(f"   image_grid_thw shape: {igt.shape}")
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    if not issues:
        print("✅ ALL CHECKS PASSED!")
        print("   Safe to proceed with training.")
    else:
        print(f"❌ FOUND {len(issues)} ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n   FIX PREPROCESSING BEFORE TRAINING!")
    
    return len(issues) == 0


if __name__ == "__main__":
    import sys
    
    paths = [
        "preprocessed_data_v2/train/preprocessed_data.pt",
        "preprocessed_data_v2/val/preprocessed_data.pt",
    ]
    
    all_passed = True
    for path in paths:
        try:
            passed = verify_preprocessed_data(path)
            all_passed = all_passed and passed
        except FileNotFoundError:
            print(f"⚠️  {path} not found - run preprocessing first")
        print("\n")
    
    if all_passed:
        print("🎉 ALL DATASETS VERIFIED - READY FOR TRAINING")
    else:
        print("⛔ FIX ISSUES BEFORE TRAINING")