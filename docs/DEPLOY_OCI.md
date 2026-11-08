# Task 7: Deploy the model API on Oracle Cloud (Pathway A)

Take a screenshot at each numbered step for the report.

The Always Free Arm allowance is now 2 OCPU and 12 GB in total for new free tenancies, not the older 4 OCPU and 24 GB. Check the limit in your own console before creating the VM. The AMD Micro shape (1 GB) is too small to load TensorFlow.

## 1. Create the VM

1. Sign in at cloud.oracle.com and open **Compute > Instances > Create instance**.
2. Name it `23L-0881_A2_Ops`.
3. Image: **Canonical Ubuntu 24.04** (the aarch64 build). Shape: **VM.Standard.A1.Flex**, 2 OCPU and 12 GB.
4. Use a public subnet with a public IPv4 address, and download the SSH key.
5. If you see "Out of capacity", try another availability domain or retry later. If it keeps failing, use a small x86 VM on Azure for Students or Google Cloud. The same Compose file works, because `requirements-serve.txt` picks the right TensorFlow package for the CPU type.

## 2. Open port 8000 in both firewalls

Cloud network rule: **Networking > Virtual cloud networks > your VCN > Security > your security list > Add ingress rule**.

| Field | Value |
| --- | --- |
| Source CIDR | 0.0.0.0/0 |
| IP protocol | TCP |
| Destination port | 8000 |

VM rule, over SSH:

```bash
ssh -i key.pem ubuntu@VM_IP
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save
```

## 3. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit        # then SSH in again so the group applies
docker --version && docker compose version
```

## 4. Put the code and the model on the VM

```bash
# on the VM
git clone https://github.com/FatimaHabib191/23L-0881_A2_Ops.git
cd 23L-0881_A2_Ops && mkdir -p models/champion

# on your computer
scp -i key.pem models/champion/model.keras ubuntu@VM_IP:~/23L-0881_A2_Ops/models/champion/
```

The serving code never contacts DagsHub. The champion file is copied once and mounted read-only into the container.

## 5. Configure and start the API

```bash
cd ~/23L-0881_A2_Ops
chmod a+r models/champion/model.keras
python3 -c "import secrets; print(secrets.token_urlsafe(32))"     # this is your API key
nano .env
```

`.env` on the VM needs only:

```
MODEL_VERSION=VisionGuard:v2
MODEL_THRESHOLD=<threshold from val_result.json>
MODEL_API_KEY=<the key you generated>
```

```bash
docker compose -f docker-compose.api.yml up -d --build
docker compose -f docker-compose.api.yml ps
docker compose -f docker-compose.api.yml logs --tail=20
```

The Arm build takes several minutes. A slow first `/health` response is normal while TensorFlow starts.

## 6. Test from another network

Turn off Wi-Fi and use a phone hotspot, or use a phone browser on mobile data.

```bash
curl http://VM_IP:8000/health
```

Postman: import `docs/VisionGuard.postman_collection.json`, then set `base_url` and `api_key`. In each predict request, choose a file for the `image` field.

| Request | Expected |
| --- | --- |
| GET /health | 200 with status ok |
| POST /v1/predict with a key | 200 with label, confidence, model_version |
| POST /v1/predict without a key | 401 |
| POST /v1/predict with a text file | 415 |

## 7. Connect local Flask (Task 8)

Set these in your local `.env`, then run `docker compose up -d --build`:

```
INFERENCE_MODE=remote
MODEL_API_URL=http://VM_IP:8000
MODEL_API_KEY=<same key>
```

`http://localhost:5000/health` should show `"model_api_reachable": true`. Send several images through the page, or run `python scripts/send_batch.py path/to/images`, and then open **History**. The Inference column shows `remote`.

## Notes for the report

- Oracle can reclaim Always Free instances it classes as idle. Check the current rule on the Always Free resources page, and keep the VM running until grading.
- The API uses plain HTTP, so the key is not encrypted in transit. That is a known limit of this coursework deployment. A real service would put HTTPS in front of it.
