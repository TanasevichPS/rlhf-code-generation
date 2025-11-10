#!/usr/bin/env python3
"""
Simple syntax test for our fixes.
"""

def test_syntax():
    """Test that our modified files have correct syntax."""
    import ast

    files_to_test = [
        'modern_rlhf/config.py',
        'modern_rlhf/reward_model.py',
        'modern_rlhf/trainer.py',
        'modern_rlhf/pipeline.py',
        'run_modern_rlhf.py',
        'quick_diagnose.py',
        'diagnose.py'
    ]

    print("Testing syntax of modified files...")

    for file_path in files_to_test:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse the AST to check syntax
            ast.parse(source)
            print(f"✅ {file_path} - syntax OK")

        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            return False
        except FileNotFoundError:
            print(f"⚠️  {file_path} - file not found")
        except Exception as e:
            print(f"❌ {file_path} - error: {e}")
            return False

    print("\n🎉 All syntax checks passed!")
    return True

if __name__ == "__main__":
    test_syntax()
