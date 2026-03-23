// Dashboard — animations au chargement
document.addEventListener("DOMContentLoaded", () => {
    // Animate KPI values counting up
    document.querySelectorAll(".kpi-value").forEach(el => {
        const text = el.textContent.trim();
        const num = parseInt(text.replace(/\D/g, ""), 10);
        if (!isNaN(num) && num > 0) {
            let start = 0;
            const duration = 1200;
            const step = Math.ceil(num / (duration / 16));
            const timer = setInterval(() => {
                start = Math.min(start + step, num);
                el.textContent = start.toLocaleString("fr-FR");
                if (start >= num) clearInterval(timer);
            }, 16);
        }
    });

    // Fade-in les cartes
    const cards = document.querySelectorAll(".kpi-card, .chart-card");
    cards.forEach((card, i) => {
        card.style.opacity = "0";
        card.style.transform = "translateY(24px)";
        setTimeout(() => {
            card.style.transition = "opacity .5s ease, transform .5s ease";
            card.style.opacity = "1";
            card.style.transform = "translateY(0)";
        }, i * 60);
    });
});
