const log = document.getElementById("log");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

function addMessage(who, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${who}`;

  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who === "user" ? "you" : "bot";

  const bubble = document.createElement("p");
  bubble.textContent = text;

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return wrap;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;

  const thinking = addMessage("bot", "...");
  thinking.classList.add("thinking");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    thinking.querySelector("p").textContent = data.reply;
    thinking.classList.remove("thinking");
  } catch (err) {
    thinking.querySelector("p").textContent =
      "Couldn't reach the server. Is it still running?";
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
});
