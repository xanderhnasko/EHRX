# Google Cloud Platform Setup for PDF2EHR VLM Integration

This guide walks through setting up Google Cloud Platform (GCP) for PDF2EHR's Vision-Language Model integration using Gemini 1.5 Flash.

## Prerequisites

- Google account
- Credit card (required for GCP, but free tier available)
- Command line access (Terminal on macOS/Linux, PowerShell on Windows)

## Step 1: Create Google Cloud Project

1. **Navigate to Google Cloud Console**
   - Go to https://console.cloud.google.com/
   - Sign in with your Google account

2. **Create New Project**
   - Click the project dropdown at the top of the page
   - Click "New Project"
   - Enter project details:
     - **Project name**: `pdf2ehr-vlm` (or your preferred name)
     - **Organization**: Leave as "No organization" unless you have one
   - Click "Create"
   - Wait for project creation (takes ~30 seconds)

3. **Select Your Project**
   - Click the project dropdown again
   - Select your newly created `pdf2ehr-vlm` project

## Step 2: Enable Required APIs

1. **Enable Vertex AI API**
   - In the Cloud Console, navigate to "APIs & Services" > "Library"
   - Search for "Vertex AI API"
   - Click on "Vertex AI API"
   - Click "Enable"
   - Wait for enablement (takes ~1-2 minutes)

2. **Verify API Enablement**
   - Navigate to "APIs & Services" > "Dashboard"
   - You should see "Vertex AI API" listed under "Enabled APIs"

## Step 3: Set Up Billing

**Note**: Gemini 1.5 Flash has generous free tier limits, but billing must be enabled.

1. **Navigate to Billing**
   - Click the menu (☰) > "Billing"
   - If prompted, click "Link a billing account" or "Create billing account"

2. **Add Payment Method**
   - Follow the prompts to add a credit card
   - Review and accept terms of service

3. **Link to Project**
   - Ensure your `pdf2ehr-vlm` project is linked to the billing account
   - Navigate to "Billing" > "Account management"
   - Verify project appears under "Projects linked to this billing account"

**Free Tier Limits** (as of 2025):
- Gemini 1.5 Flash: 1,500 requests/day free
- First 60 API calls/minute free
- See current pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing

## Step 4: Create Service Account

Service accounts allow PDF2EHR to authenticate with Google Cloud APIs.

1. **Navigate to IAM & Admin**
   - Click menu (☰) > "IAM & Admin" > "Service Accounts"

2. **Create Service Account**
   - Click "Create Service Account"
   - Enter details:
     - **Service account name**: `pdf2ehr-vlm-sa`
     - **Service account ID**: `pdf2ehr-vlm-sa` (auto-filled)
     - **Description**: "Service account for PDF2EHR VLM processing"
   - Click "Create and Continue"

3. **Grant Permissions**
   - In "Grant this service account access to project"
   - Select role: **"Vertex AI User"**
   - Click "Continue"
   - Click "Done"

## Step 5: Create and Download Service Account Key

1. **Navigate to Service Account**
   - In "Service Accounts" list, find `pdf2ehr-vlm-sa@[YOUR-PROJECT-ID].iam.gserviceaccount.com`
   - Click the three dots (⋮) under "Actions"
   - Select "Manage keys"

2. **Create JSON Key**
   - Click "Add Key" > "Create new key"
   - Select "JSON" format
   - Click "Create"
   - A JSON file will automatically download to your computer

3. **Secure the Key File**
   - **IMPORTANT**: This file grants access to your GCP project. Keep it secure!
   - Rename the file to something memorable: `pdf2ehr-gcp-credentials.json`
   - Move to a secure location on your system (e.g., `~/.config/gcp/`)

   ```bash
   # Example: Create secure directory and move key
   mkdir -p ~/.config/gcp
   mv ~/Downloads/pdf2ehr-vlm-sa-*.json ~/.config/gcp/pdf2ehr-credentials.json
   chmod 600 ~/.config/gcp/pdf2ehr-credentials.json
   ```

4. **Never Commit to Git**
   - Ensure `.gitignore` includes credential paths:
   ```
   # GCP credentials
   *-credentials.json
   *.json
   .config/
   ```

## Step 6: Configure Environment Variables

PDF2EHR uses environment variables to locate GCP credentials.

### Option A: Per-Session (Temporary)

```bash
# macOS/Linux
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp/pdf2ehr-credentials.json"
export GCP_PROJECT_ID="pdf2ehr-vlm"  # Use your actual project ID
export GCP_LOCATION="us-central1"

# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\YourName\.config\gcp\pdf2ehr-credentials.json"
$env:GCP_PROJECT_ID="pdf2ehr-vlm"
$env:GCP_LOCATION="us-central1"
```

### Option B: Persistent Configuration

**macOS/Linux** (bash/zsh):
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp/pdf2ehr-credentials.json"' >> ~/.zshrc
echo 'export GCP_PROJECT_ID="pdf2ehr-vlm"' >> ~/.zshrc
echo 'export GCP_LOCATION="us-central1"' >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc
```

**Windows PowerShell**:
```powershell
# Add to PowerShell profile
Add-Content $PROFILE "`n`$env:GOOGLE_APPLICATION_CREDENTIALS='C:\Users\YourName\.config\gcp\pdf2ehr-credentials.json'"
Add-Content $PROFILE "`$env:GCP_PROJECT_ID='pdf2ehr-vlm'"
Add-Content $PROFILE "`$env:GCP_LOCATION='us-central1'"
```

### Option C: Using .env File (Recommended for Development)

Create a `.env` file in the PDF2EHR project root:

```bash
# .env
GOOGLE_APPLICATION_CREDENTIALS=/Users/YourName/.config/gcp/pdf2ehr-credentials.json
GCP_PROJECT_ID=pdf2ehr-vlm
GCP_LOCATION=us-central1
```

**Important**: Add `.env` to `.gitignore`!

PDF2EHR will automatically load `.env` using `python-dotenv` if available.

## Step 7: Install Required Python Packages

```bash
# Navigate to PDF2EHR directory
cd /path/to/PDF2EHR

# Install Google Cloud dependencies
pip install google-cloud-aiplatform>=1.38.0 python-dotenv>=1.0.0

# Or install all dependencies (if requirements.txt updated)
pip install -r requirements.txt
```

## Step 8: Verify Setup

Test your GCP configuration with this Python script:

```python
# test_gcp_setup.py
import os
from google.cloud import aiplatform

# Check environment variables
print("Checking environment variables...")
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
project_id = os.getenv("GCP_PROJECT_ID")
location = os.getenv("GCP_LOCATION", "us-central1")

print(f"  ✓ Credentials: {credentials_path}")
print(f"  ✓ Project ID: {project_id}")
print(f"  ✓ Location: {location}")

# Verify credentials file exists
if not os.path.exists(credentials_path):
    print(f"  ✗ ERROR: Credentials file not found at {credentials_path}")
    exit(1)
print(f"  ✓ Credentials file exists")

# Initialize Vertex AI
print("\nInitializing Vertex AI...")
aiplatform.init(project=project_id, location=location)
print("  ✓ Vertex AI initialized successfully")

print("\n✅ GCP setup verified! Ready to use Gemini models.")
```

Run the test:
```bash
python test_gcp_setup.py
```

Expected output:
```
Checking environment variables...
  ✓ Credentials: /Users/YourName/.config/gcp/pdf2ehr-credentials.json
  ✓ Project ID: pdf2ehr-vlm
  ✓ Location: us-central1
  ✓ Credentials file exists

Initializing Vertex AI...
  ✓ Vertex AI initialized successfully

✅ GCP setup verified! Ready to use Gemini models.
```

## Step 9: Configuration in PDF2EHR

Update `configs/default.yaml` with your GCP settings:

```yaml
vlm:
  project_id: "pdf2ehr-vlm"  # Your GCP project ID
  location: "us-central1"
  model_name: "gemini-1.5-flash"
  max_tokens: 8192
  temperature: 0.1
  confidence_threshold: 0.85
```

## Troubleshooting

### Error: "Could not automatically determine credentials"

**Cause**: `GOOGLE_APPLICATION_CREDENTIALS` not set or pointing to wrong file.

**Solution**:
```bash
# Verify environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS  # macOS/Linux
echo $env:GOOGLE_APPLICATION_CREDENTIALS  # Windows

# Re-export if needed
export GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/credentials.json"
```

### Error: "Permission denied" when calling Vertex AI

**Cause**: Service account lacks "Vertex AI User" role.

**Solution**:
1. Go to IAM & Admin > IAM
2. Find your service account (`pdf2ehr-vlm-sa@...`)
3. Click "Edit principal" (pencil icon)
4. Add role "Vertex AI User"
5. Save

### Error: "Vertex AI API has not been used in project before"

**Cause**: Vertex AI API not enabled.

**Solution**:
1. Navigate to APIs & Services > Library
2. Search "Vertex AI API"
3. Click "Enable"
4. Wait 1-2 minutes for propagation

### Error: "Quota exceeded"

**Cause**: Exceeded free tier limits.

**Solution**:
- Check Vertex AI quotas in Cloud Console > IAM & Admin > Quotas
- Wait for quota reset (daily for free tier)
- Consider upgrading billing if needed for production use

## Security Best Practices

1. **Never commit credentials to Git**
   - Always use `.gitignore` for credential files
   - Use environment variables or secret management

2. **Rotate keys periodically**
   - Create new service account keys every 90 days
   - Delete old keys from GCP Console

3. **Use least privilege**
   - Only grant "Vertex AI User" role (not Owner or Editor)
   - Create separate service accounts for different environments

4. **Monitor usage**
   - Check Cloud Console billing regularly
   - Set up budget alerts (Billing > Budgets & Alerts)

## Cost Monitoring

Set up budget alerts to avoid unexpected charges:

1. Navigate to "Billing" > "Budgets & alerts"
2. Click "Create budget"
3. Configure:
   - **Name**: "PDF2EHR Monthly Budget"
   - **Projects**: Select `pdf2ehr-vlm`
   - **Budget amount**: $10 (or your preferred limit)
   - **Alert thresholds**: 50%, 90%, 100%
4. Add email for notifications
5. Click "Finish"

## Next Steps

Once GCP is configured:

1. Run `test_gcp_setup.py` to verify everything works
2. Update `configs/default.yaml` with your project settings
3. Start using VLM features in PDF2EHR

## Useful Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [Service Account Best Practices](https://cloud.google.com/iam/docs/best-practices-service-accounts)
- [Vertex AI Python SDK Reference](https://cloud.google.com/python/docs/reference/aiplatform/latest)

## Support

For issues with:
- **GCP setup**: Check [GCP Support](https://cloud.google.com/support)
- **PDF2EHR integration**: Open issue on GitHub
- **Billing questions**: Contact GCP Billing Support
