# Running the Full MVP Pipeline on SENSITIVE_ehr2_copy.pdf

## Quick Start

```bash
python3 run_full_ehr2.py
```

The script will:
1. ✅ Automatically find `SENSITIVE_ehr2_copy.pdf` in the current directory
2. ✅ Ask you for the page range to process
3. ✅ Process the document using VLM extraction
4. ✅ Group pages into sub-documents
5. ✅ Initialize an interactive query agent
6. ✅ Let you ask natural language questions about the EHR

## What Happens

### Stage 1: Multi-Page VLM Extraction
- Processes each page using Google's Gemini Vision model
- Extracts structured data (text blocks, tables, charts, etc.)
- Saves checkpoints every 50 pages
- Tracks processing time and API costs

### Stage 2: Sub-Document Grouping
- Groups consecutive pages into logical sub-documents
- Detects document types (e.g., "Lab Results", "Medication List")
- Creates hierarchical index
- Saves enhanced schema

### Stage 3: Hybrid Query Agent
- Initializes intelligent query agent
- Combines schema filtering with VLM reasoning
- Ready to answer questions

### Stage 4: Interactive Query Mode
- Ask questions in natural language
- Get structured answers with matched elements
- See reasoning and confidence scores

## Example Queries

Once in interactive mode, you can ask:

```
💬 Query: What medications is the patient taking?
💬 Query: What were the blood test results?
💬 Query: Show me all vital signs
💬 Query: What is the patient's diagnosis?
💬 Query: When was the patient admitted?
💬 Query: List all allergies
💬 Query: What procedures were performed?
```

## Special Commands

- `stats` - Show document statistics
- `examples` - Run pre-defined example queries
- `quit` or `exit` - Exit query mode

## Output Files

All outputs are saved to `output/ehr2_full/`:

1. **`SENSITIVE_ehr2_copy_full.json`** - Complete extraction with all elements
2. **`SENSITIVE_ehr2_copy_enhanced.json`** - Enhanced with sub-document grouping
3. **`SENSITIVE_ehr2_copy_index.json`** - Hierarchical index for navigation
4. **`ehr2_pipeline.log`** - Detailed processing log

## Processing Options

### Process Full Document (All Pages)
```bash
python3 run_full_ehr2.py
# When prompted, enter: all
```

### Process Specific Pages
```bash
python3 run_full_ehr2.py
# When prompted, enter: 1-20
# Or: 1-10,15-20,25
```

### Test Mode (First 5 Pages)
```bash
python3 run_full_ehr2.py
# When prompted, enter: 1-5
```

## Requirements

Make sure you have:
- ✅ `.env` file with GCP credentials
- ✅ `SENSITIVE_ehr2_copy.pdf` in the current directory
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Network access for Gemini API calls

## Estimated Processing Time

- **Per page**: ~2-5 seconds (depending on complexity)
- **20 pages**: ~1-2 minutes
- **100 pages**: ~5-10 minutes
- **Full document (650 pages)**: ~30-60 minutes

## Estimated Costs

Using Gemini 1.5 Flash:
- **Per page**: ~$0.001-0.002
- **20 pages**: ~$0.02-0.04
- **100 pages**: ~$0.10-0.20
- **Full document (650 pages)**: ~$0.65-1.30

## Troubleshooting

### "GCP project_id must be provided"
- Check your `.env` file has `GCP_PROJECT_ID` set
- Verify `GOOGLE_APPLICATION_CREDENTIALS` points to valid credentials

### "PDF not found"
- Ensure `SENSITIVE_ehr2_copy.pdf` is in `/Users/justinjasper/PDF2EHR/`
- Or provide full path when running

### "API rate limit exceeded"
- The pipeline includes automatic retry logic
- Processing will continue after brief delays
- Check `ehr2_pipeline.log` for details

### Out of memory
- Process in smaller batches (e.g., 50 pages at a time)
- Reduce DPI in the script (change `dpi=200` to `dpi=150`)

## Advanced Usage

### Use Original run_mvp_pipeline.py
For more control:
```bash
python3 run_mvp_pipeline.py
# Then enter path: SENSITIVE_ehr2_copy.pdf
# Then enter range: all
```

### Process Multiple Documents
Run the script multiple times with different PDFs, or modify the script to loop through multiple files.

## Next Steps

After processing:
1. Review the generated JSON files in `output/ehr2_full/`
2. Use the interactive query mode to explore the data
3. Export specific queries to JSON for downstream processing
4. Integrate with your EHR UI or analytics pipeline

## Support

Check the logs:
```bash
tail -f ehr2_pipeline.log
```

For issues, refer to:
- `HIERARCHY_IMPLEMENTATION.md` - Hierarchy structure details
- `HIERARCHY_FIXES.md` - Recent fixes and changes
- Pipeline logs in `ehr2_pipeline.log`




