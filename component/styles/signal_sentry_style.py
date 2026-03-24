"""SignalSentry-specific style layer."""

from component.styles.main_style import MAIN_STYLE

SIGNAL_SENTRY_STYLE = MAIN_STYLE + """
QLabel[class='brand'] {
    font-size: 24px;
    font-weight: 700;
    color: #f7f9ff;
}
QLabel[class='badge'] {
    background-color: #1a2550;
    border: 1px solid #2d3c74;
    border-radius: 10px;
    color: #d5ddff;
    padding: 2px 8px;
    font-size: 11px;
}
QLabel[class='criticalPill'] {
    background-color: #3f1111;
    color: #fecaca;
    border: 1px solid #7f1d1d;
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}
QLabel[class='highPill'] {
    background-color: #3f2a0b;
    color: #fde68a;
    border: 1px solid #854d0e;
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}
QLabel[class='mediumPill'] {
    background-color: #0d214f;
    color: #bfdbfe;
    border: 1px solid #1e3a8a;
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}
QLabel[class='agentChip'] {
    background-color: #eef2ff;
    color: #1d2c56;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}
QFrame[class='alertRow'] {
    background-color: #121a33;
    border: 1px solid #2b375f;
    border-radius: 12px;
}
QProgressBar {
    border: 1px solid #2a3763;
    border-radius: 8px;
    background: #0e1530;
    text-align: center;
}
QProgressBar::chunk {
    border-radius: 7px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #21c55d, stop:1 #86efac);
}
QProgressBar[class='warn']::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #f59e0b, stop:1 #fcd34d);
}
"""
