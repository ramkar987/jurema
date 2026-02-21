"""Funções auxiliares para formatação e validação."""

import re


def format_confidence(score: float) -> str:
    """Retorna emoji + texto de acordo com o nível de confiança."""
    if score >= 80:
        return f"🟢 {score:.0f}% — Alta"
    elif score >= 55:
        return f"🟡 {score:.0f}% — Média"
    else:
        return f"🔴 {score:.0f}% — Baixa"


def format_elapsed(seconds: float) -> str:
    """Formata tempo de execução de forma legível."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.1f} s"


def truncar(texto: str, max_chars: int = 250) -> str:
    """Trunca texto longo preservando palavras completas."""
    if len(texto) <= max_chars:
        return texto
    return texto[:max_chars].rsplit(" ", 1)[0] + "…"


def sanitizar_nome(nome: str) -> str:
    """Remove caracteres especiais de nomes de arquivo."""
    return re.sub(r"[^\w\-_\. ]", "_", nome)
