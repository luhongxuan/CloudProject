import plotly.graph_objects as go

# 假設這是 XGBoost 輸出的預測機率
qs_prob = 0.82  # 82% 優質先發機率

fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = qs_prob * 100,
    delta = {'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
    title = {'text': "Quality Start Probability (%)", 'font': {'size': 24}},
    gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 40], 'color': '#ff4d4d'},    # 紅色：低機率
            {'range': [40, 70], 'color': '#ffcc00'},   # 黃色：中機率
            {'range': [70, 100], 'color': '#00cc66'}   # 綠色：高機率
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': qs_prob * 100
        }
    }
))

fig.update_layout(
    paper_bgcolor="white",
    font={'color': "black", 'family': "Arial"}
)

fig.show()
