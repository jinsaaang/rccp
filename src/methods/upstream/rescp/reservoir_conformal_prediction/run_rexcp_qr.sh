#!/bin/bash

export PYTHONPATH="/path/to/reservoir-conformal-prediction-dev-main/:$PYTHONPATH"

# Beijing
python run_RExCP_qr.py src_dir="/path/to/logs/base/beijing/rnn/2025-07-16/17-50-09/" dataset=beijing add_exogenous=True conformal_predictor=rexcp_qr_beijing_rnn_emb
python run_RExCP_qr.py src_dir="/path/to/logs/base/beijing/arima/2025-07-24/04-12-06/" dataset=beijing add_exogenous=True conformal_predictor=rexcp_qr_beijing_arima_emb
python run_RExCP_qr.py src_dir="/path/to/logs/base/beijing/transformer/2025-07-22/16-19-56/" dataset=beijing add_exogenous=True conformal_predictor=rexcp_qr_beijing_transformer_emb

# Solar
python run_RExCP_qr.py src_dir="/path/to/logs/base/solar/rnn/2025-07-16/17-55-58/" dataset=solar add_exogenous=True conformal_predictor=rexcp_qr_solar_rnn_emb
python run_RExCP_qr.py src_dir="/path/to/logs/base/solar/arima/2025-07-27/14-47-17/" dataset=solar add_exogenous=True conformal_predictor=rexcp_qr_solar_arima_emb
python run_RExCP_qr.py src_dir="/path/to/logs/base/solar/transformer/2025-07-22/16-22-57/" dataset=solar add_exogenous=True conformal_predictor=rexcp_qr_solar_transformer_emb

# ACEA
python run_RExCP_qr.py src_dir="/path/to/logs/base/elec/rnn/2025-09-01/11-44-56" dataset=elec add_exogenous=True conformal_predictor=rexcp_qr_elec_rnn
python run_RExCP_qr.py src_dir="/path/to/logs/base/elec/arima/2025-09-01/11-57-39" dataset=elec add_exogenous=True conformal_predictor=rexcp_qr_elec_arima
python run_RExCP_qr.py src_dir="/path/to/logs/base/elec/transformer/2025-09-01/11-54-31" dataset=elec add_exogenous=True conformal_predictor=rexcp_qr_elec_transformer

# Exchange
python run_RExCP_qr.py src_dir="/path/to/logs/base/exchange/rnn/2025-08-25/14-38-06" dataset=exchange add_exogenous=True conformal_predictor=rexcp_qr_exchange_rnn_delay7
python run_RExCP_qr.py src_dir="/path/to/logs/base/exchange/arima/2025-08-25/14-42-35" dataset=exchange add_exogenous=True conformal_predictor=rexcp_qr_exchange_arima_delay7
python run_RExCP_qr.py src_dir="/path/to/logs/base/exchange/transformer/2025-08-25/14-40-48" dataset=exchange add_exogenous=True conformal_predictor=rexcp_qr_exchange_transformer_delay7