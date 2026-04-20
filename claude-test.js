import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function run() {
  const res = await client.messages.create({
    model: "claude-3-haiku-20240307",
    max_tokens: 300,
    messages: [
      { role: "user", content: "Explain EduAIPlatform in 2 lines" }
    ],
  });

  console.log(res.content[0].text);
}

run();