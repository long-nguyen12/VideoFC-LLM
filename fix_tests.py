import os
import re

def fix_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Classes to replace with dict
    classes = [
        'ClaimDecomposition', 'EvidenceRef', 'EvidenceStrengthReport',
        'FinalVerdict', 'HopResult', 'ModalConflictReport', 'ReasoningStep',
        'SubQuestion', 'VideoSegment', 'ExplainabilityReport', 'PipelineInputs'
    ]
    for cls in classes:
        content = re.sub(rf'\b{cls}\b(?=\()', 'dict', content)
        
    # Replace from schemas.data_models import ...
    content = re.sub(r'from schemas\.data_models import \([^)]+\)', '', content, flags=re.DOTALL)
    content = re.sub(r'from schemas\.data_models import[^\n]*\n', '', content)

    # Some tests check isinstance(report, ExplainabilityReport). Replace with isinstance(report, dict)
    content = re.sub(r'isinstance\(([^,]+),\s*ExplainabilityReport\)', r'isinstance(\1, dict)', content)

    # Replace result.property with result['property']
    # Common variables
    vars_to_fix = [
        'result', 'results\[0\]', 'results\[1\]', 'verdict', 'report', 's', 'hop', 'sq',
        'claim_decomp', 'modal_report', 'segment', 'strength_pass', 'strength_fail',
        'hop_saliencies\[hop\[\"hop\"\]\]', 'inputs', 'annotations\[0\]'
    ]
    for var in vars_to_fix:
        # We need to catch var.PROPERTY, e.g. result.claim_id
        # Note: var might have regex special characters like \[0\]
        # So we just construct a string replacement regex
        content = re.sub(rf'\b({var})\.([a-z_][a-z0-9_]*)\b', r'\1["\2"]', content)

    # Also fix properties of elements in lists, e.g. results[0].answer_unknown
    content = re.sub(r'(\b[a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\])\.([a-z_][a-z0-9_]*)\b', r'\1["\2"]', content)
    
    # Fix dict(hop=1) syntax used by mock classes which we just changed to dict(...) -> dict(hop=1) is valid python :)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

for path in ['tests/test_modules.py', 'tests/test_dataset.py', 'tests/test_single_llm.py']:
    if os.path.exists(path):
        fix_file(path)
