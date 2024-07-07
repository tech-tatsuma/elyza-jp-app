# Llama-3-ELYZA-JP-8B-app
このレポジトリはELYZA社から提供されているLlama-3-ELYZA-JPモデルの軽量8Bバージョンを使ったwebアプリをローカルで立ち上げるためのものです．
## フォルダー構造
```
- app.py                　　　　　　　　　　　　　　　　　　　　　
- docker-compose.yaml   　　　　　　　　　　　　　　　　　　　　　
- Dockerfile　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
- Llama-3-ELYZA-JP-8B-q4_k_m.gguf 　// インストールしてきたLlamaモデル
- requirements.txt      　　　　　　　　　　　　　　　　　　　　
```
## 実行手順
### Docker環境を使わない場合
```
wget "https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF/resolve/main/Llama-3-ELYZA-JP-8B-q4_k_m.gguf?download=true"
virtualenv llmenv
source llmenv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8080
```
### Docker環境を使う場合
```
wget "https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF/resolve/main/Llama-3-ELYZA-JP-8B-q4_k_m.gguf?download=true"
docker compose up
```

## 機能
- Simple Chat：シンプルなモデルとのチャット機能．
- PDFRAG機能：PDFをアップロードすることでそのドキュメントに基づいた回答を行う．
- Agent機能：最新の情報をwebから取得し，それに基づいた回答を行う機能．