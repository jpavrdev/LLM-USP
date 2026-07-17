const $ = (sel) => document.querySelector(sel);

const messagesEl = $("#messages");
const form = $("#composer-form");
const input = $("#input");
const sendBtn = $("#send");
const themeBtn = $("#theme-toggle");
const newChatBtn = $("#new-chat");
const historyEl = $("#history-list");

const tempSlider = $("#temperature");
const topkSlider = $("#top_k");
const maxSlider = $("#max_new_tokens");
const tempVal = $("#temp-val");
const topkVal = $("#topk-val");
const maxVal = $("#max-val");

const modelInfoEl = $("#model-info");
const deviceBadge = $("#device-badge");
const deviceText = $("#device-text");

let streaming = false;
let abortCtrl = null;

// Histórico de conversas (só na memória da aba — sem persistência)
const conversations = [];
let currentConv = null;

// ---------- Tema ----------
const savedTheme = localStorage.getItem("theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
document.documentElement.dataset.theme = savedTheme;
themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
});

// ---------- Sliders ----------
const syncSliderLabels = () => {
    tempVal.textContent = parseFloat(tempSlider.value).toFixed(2);
    topkVal.textContent = topkSlider.value === "0" ? "off" : topkSlider.value;
    maxVal.textContent = maxSlider.value;
};
[tempSlider, topkSlider, maxSlider].forEach((s) => s.addEventListener("input", syncSliderLabels));
syncSliderLabels();

// ---------- Auto-resize do textarea ----------
const autoResize = () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 240) + "px";
};
input.addEventListener("input", autoResize);

input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
    }
});

// ---------- Info do modelo ----------
async function loadInfo() {
    try {
        const res = await fetch("/api/info");
        const data = await res.json();
        modelInfoEl.textContent = `${data.params_millions}M params · ${data.device}`;
        deviceBadge.textContent = data.device.toUpperCase();
        deviceText.textContent = data.device === "cuda" ? "na GPU" : "na CPU";
    } catch (err) {
        modelInfoEl.textContent = "offline";
    }
}
loadInfo();

// ---------- Conversas ----------
function createConversation() {
    const conv = {
        id: crypto.randomUUID(),
        title: "Nova conversa",
        messages: [],
    };
    conversations.unshift(conv);
    renderHistory();
    setCurrent(conv);
    return conv;
}

function setCurrent(conv) {
    currentConv = conv;
    renderHistory();
    renderMessages();
    updateTitle();
}

function updateTitle() {
    $(".chat-title").textContent = currentConv?.title || "Nova conversa";
}

function renderHistory() {
    historyEl.innerHTML = "";
    conversations.forEach((c) => {
        const el = document.createElement("div");
        el.className = "history-item" + (c === currentConv ? " active" : "");
        el.textContent = c.title;
        el.addEventListener("click", () => setCurrent(c));
        historyEl.appendChild(el);
    });
}

function renderMessages() {
    messagesEl.innerHTML = "";
    if (!currentConv || currentConv.messages.length === 0) {
        renderWelcome();
        return;
    }
    for (const msg of currentConv.messages) appendMessageEl(msg);
    scrollToBottom();
}

function renderWelcome() {
    messagesEl.innerHTML = `
        <div class="welcome">
            <div class="welcome-mark">
                <svg viewBox="0 0 64 64" width="56" height="56" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="32" cy="32" r="28"/>
                    <path d="M22 26c3-4 6-6 10-6s7 2 10 6"/>
                    <path d="M22 38c3 4 6 6 10 6s7-2 10-6"/>
                </svg>
            </div>
            <h1>Olá — eu sou sua LLM local</h1>
            <p>Pergunte algo em português. Modelo Tucano-1b1-Instruct (PUCRS, 1.1B parâmetros).</p>
            <div class="suggestions">
                <button class="suggestion" data-prompt="O que é inteligência artificial?">O que é inteligência artificial?</button>
                <button class="suggestion" data-prompt="Como funciona a fotossíntese?">Como funciona a fotossíntese?</button>
                <button class="suggestion" data-prompt="Escreva uma breve história sobre um astronauta que descobre um planeta habitado.">Escreva uma breve história sobre um astronauta…</button>
                <button class="suggestion" data-prompt="Liste três dicas práticas para melhorar o foco no trabalho.">Liste três dicas para melhorar o foco no trabalho</button>
            </div>
        </div>`;
    bindSuggestions();
}

function bindSuggestions() {
    document.querySelectorAll(".suggestion").forEach((btn) => {
        btn.addEventListener("click", () => {
            input.value = btn.dataset.prompt;
            autoResize();
            input.focus();
        });
    });
}
bindSuggestions();

// ---------- Renderização de mensagens ----------
function buildSourcesEl(sources) {
    const el = document.createElement("div");
    el.className = "sources";
    const label = document.createElement("span");
    label.className = "sources-label";
    label.textContent = "Fontes (Wikipedia):";
    el.appendChild(label);
    sources.forEach((s) => {
        const a = document.createElement(s.url ? "a" : "span");
        a.className = "source-link";
        a.textContent = s.title;
        if (s.url) {
            a.href = s.url;
            a.target = "_blank";
            a.rel = "noopener";
        }
        el.appendChild(a);
    });
    return el;
}

function appendMessageEl(msg) {
    const wrap = document.createElement("div");
    wrap.className = `message ${msg.role}`;
    wrap.dataset.id = msg.id;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    if (msg.role === "assistant") {
        avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/></svg>`;
    }

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const author = document.createElement("div");
    author.className = "bubble-author";
    author.textContent = msg.role === "user" ? "Você" : "GPT PT-BR";

    const body = document.createElement("div");
    body.className = "bubble-body";
    body.textContent = msg.content;
    if (msg.sources && msg.sources.length) {
        body.appendChild(buildSourcesEl(msg.sources));
    }

    bubble.appendChild(author);
    bubble.appendChild(body);
    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    return { wrap, body };
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ---------- Envio ----------
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (streaming) {
        abortCtrl?.abort();
        return;
    }

    const prompt = input.value.trim();
    if (!prompt) return;

    if (!currentConv) createConversation();
    else if (currentConv.messages.length === 0) {
        // primeira mensagem — limpa o welcome
        messagesEl.innerHTML = "";
    }

    // Define título da conversa com base no primeiro prompt
    if (currentConv.messages.length === 0) {
        currentConv.title = prompt.slice(0, 40) + (prompt.length > 40 ? "…" : "");
        updateTitle();
        renderHistory();
    }

    const userMsg = { id: crypto.randomUUID(), role: "user", content: prompt };
    currentConv.messages.push(userMsg);
    appendMessageEl(userMsg);

    input.value = "";
    autoResize();
    scrollToBottom();

    const assistantMsg = { id: crypto.randomUUID(), role: "assistant", content: "" };
    currentConv.messages.push(assistantMsg);
    const { body } = appendMessageEl(assistantMsg);
    body.innerHTML = '<span class="thinking"><span></span><span></span><span></span></span>';
    scrollToBottom();

    await streamCompletion(prompt, assistantMsg, body);
});

async function streamCompletion(prompt, assistantMsg, bodyEl) {
    setStreaming(true);
    abortCtrl = new AbortController();

    // Para um modelo autoregressivo puro de continuação, o contexto natural é o
    // histórico concatenado. Aqui enviamos só o último prompt do usuário — simples
    // e alinhado com o que o modelo v6 foi treinado (completion de texto).
    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt,
                temperature: parseFloat(tempSlider.value),
                top_k: parseInt(topkSlider.value, 10) || null,
                max_new_tokens: parseInt(maxSlider.value, 10),
            }),
            signal: abortCtrl.signal,
        });

        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

        bodyEl.innerHTML = '<span class="stream-text"></span><span class="cursor"></span>';
        const streamText = bodyEl.querySelector(".stream-text");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let acc = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let idx;
            while ((idx = buffer.indexOf("\n\n")) !== -1) {
                const frame = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                const line = frame.trim();
                if (!line.startsWith("data:")) continue;

                const payload = line.slice(5).trim();
                if (payload === "[DONE]") break;

                try {
                    const data = JSON.parse(payload);
                    if (data.sources) {
                        assistantMsg.sources = data.sources;
                        if (!bodyEl.querySelector(".sources")) {
                            bodyEl.insertBefore(buildSourcesEl(data.sources), bodyEl.firstChild);
                        }
                        scrollToBottom();
                    } else if (data.delta) {
                        acc += data.delta;
                        streamText.textContent = acc;
                        scrollToBottom();
                    }
                } catch { /* ignora frame malformado */ }
            }
        }

        assistantMsg.content = acc;
        bodyEl.querySelector(".cursor")?.remove();
    } catch (err) {
        if (err.name === "AbortError") {
            const acc = bodyEl.querySelector(".stream-text")?.textContent || "";
            assistantMsg.content = acc;
            bodyEl.querySelector(".cursor")?.remove();
            bodyEl.querySelector(".stream-text").textContent = acc + " ⏹";
        } else {
            bodyEl.textContent = `Erro: ${err.message}`;
            assistantMsg.content = `[erro] ${err.message}`;
        }
    } finally {
        setStreaming(false);
        abortCtrl = null;
    }
}

function setStreaming(v) {
    streaming = v;
    sendBtn.classList.toggle("streaming", v);
    sendBtn.title = v ? "Parar" : "Enviar";
}

// ---------- Nova conversa ----------
newChatBtn.addEventListener("click", () => {
    createConversation();
    input.focus();
});

// ---------- Boot ----------
createConversation();
input.focus();
