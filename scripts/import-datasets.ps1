# Import the training mix picked from Hugging Face's most-downloaded instruction datasets (November 2026).
#
# Valta's fine-tuned 3B handles personal conversation, and the training gate scores it on code, so the mix is
# mostly verified, complete Python answers plus some real conversation. Every import streams the dataset and
# stops once --limit new conversations are kept; nothing is downloaded in full. Conversations already stored
# are skipped, so re-running adds up to --limit more from each. Datasets without a langdetect field are
# checked for English prose (OpenAssistant in tulu-3 is mostly other languages).
#
#   KodCode-V1-SFT-4o             Python problems whose GPT-4o answers passed unit tests     (cc-by-nc-4.0)
#   self-oss-instruct-sc2         Python instructions whose answers were run and filtered    (odc-by)
#   InstructCoder                 edit this code as asked, answer is the whole changed code  (no license tag)
#   opc-sft-stage2 educational    short, tested Python functions from OpenCoder's SFT        (mit)
#   CodeFeedback-Filtered         harder multi-step coding requests, Python only             (apache-2.0)
#   Magicoder-OSS-Instruct-75K    problems seeded from open-source code, Python only         (mit)
#   tulu-3-sft-mixture            real user conversations (WildChat, OpenAssistant) only     (odc-by)

$ErrorActionPreference = "Continue"
$valta = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\valta.exe"

& $valta dataset import KodCode/KodCode-V1-SFT-4o --code-only --min-answer 200 --limit 300
& $valta dataset import bigcode/self-oss-instruct-sc2-exec-filter-50k --code-only --min-answer 200 --limit 400
& $valta dataset import likaixin/InstructCoder --code-only --min-answer 150 --limit 300
& $valta dataset import OpenCoder-LLM/opc-sft-stage2 --config educational_instruct --code-only --min-answer 150 --limit 300
& $valta dataset import m-a-p/CodeFeedback-Filtered-Instruction --where lang=python --code-only --min-answer 200 --limit 250
& $valta dataset import ise-uiuc/Magicoder-OSS-Instruct-75K --where lang=python --code-only --min-answer 200 --limit 300
& $valta dataset import allenai/tulu-3-sft-mixture --where source=wildchat --where source=oasst --min-answer 200 --limit 300
& $valta dataset list
