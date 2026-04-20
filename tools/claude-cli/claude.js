#!/usr/bin/env node

const { Command } = require("commander");
const fs = require("fs-extra");
const path = require("path");
const { execSync } = require("child_process");

const program = new Command();

const ROOT = process.cwd();
const CLAUDE_DIR = path.join(ROOT, ".claude");
const MARKETPLACE_FILE = path.join(CLAUDE_DIR, "marketplace.json");
const PLUGINS_DIR = path.join(CLAUDE_DIR, "plugins");
const INSTALLED_FILE = path.join(CLAUDE_DIR, "installed.json");

function ensureBase() {
  fs.ensureDirSync(CLAUDE_DIR);
  fs.ensureDirSync(PLUGINS_DIR);

  if (!fs.existsSync(MARKETPLACE_FILE)) {
    fs.writeJsonSync(MARKETPLACE_FILE, { repos: [] }, { spaces: 2 });
  }

  if (!fs.existsSync(INSTALLED_FILE)) {
    fs.writeJsonSync(INSTALLED_FILE, { plugins: [] }, { spaces: 2 });
  }
}

function readJson(file) {
  ensureBase();
  return fs.readJsonSync(file);
}

function writeJson(file, data) {
  ensureBase();
  fs.writeJsonSync(file, data, { spaces: 2 });
}

function repoNameFromUrl(repo) {
  const parts = repo.split("/");
  return parts[parts.length - 1].replace(/\.git$/, "");
}

function pluginPath(name) {
  return path.join(PLUGINS_DIR, name);
}

program
  .name("claude")
  .description("Local custom Claude-style plugin CLI")
  .version("1.0.0");

const plugin = program.command("plugin").description("Plugin commands");

const marketplace = plugin
  .command("marketplace")
  .description("Marketplace operations");

marketplace
  .command("add <repo>")
  .description("Add a GitHub repo to marketplace")
  .action((repo) => {
    ensureBase();
    const data = readJson(MARKETPLACE_FILE);

    if (data.repos.includes(repo)) {
      console.log(`Already added: ${repo}`);
      return;
    }

    data.repos.push(repo);
    writeJson(MARKETPLACE_FILE, data);
    console.log(`Marketplace repo added: ${repo}`);
  });

marketplace
  .command("list")
  .description("List marketplace repos")
  .action(() => {
    ensureBase();
    const data = readJson(MARKETPLACE_FILE);

    if (!data.repos.length) {
      console.log("No marketplace repos added.");
      return;
    }

    console.log("Marketplace repos:");
    data.repos.forEach((repo, i) => {
      console.log(`${i + 1}. ${repo}`);
    });
  });

plugin
  .command("install <spec>")
  .description("Install plugin using name@repoFolder")
  .action((spec) => {
    ensureBase();

    const [pluginName, repoFolder] = spec.split("@");

    if (!pluginName || !repoFolder) {
      console.error("Use format: claude plugin install <name>@<repoFolder>");
      process.exit(1);
    }

    const marketplaceData = readJson(MARKETPLACE_FILE);
    const match = marketplaceData.repos.find(
      (repo) => repoNameFromUrl(repo).toLowerCase() === repoFolder.toLowerCase()
    );

    if (!match) {
      console.error(
        `No marketplace repo found for '${repoFolder}'. Add it first with:\nclaude plugin marketplace add <owner/repo>`
      );
      process.exit(1);
    }

    const targetDir = pluginPath(pluginName);

    if (fs.existsSync(targetDir)) {
      console.log(`Plugin already installed at ${targetDir}`);
      return;
    }

    const repoUrl = `https://github.com/${match}.git`;

    console.log(`Cloning ${repoUrl} into ${targetDir}...`);
    execSync(`git clone ${repoUrl} "${targetDir}"`, { stdio: "inherit" });

    const pkgJson = path.join(targetDir, "package.json");
    const requirementsTxt = path.join(targetDir, "requirements.txt");

    if (fs.existsSync(pkgJson)) {
      console.log("Detected package.json, running npm install...");
      execSync("npm install", { cwd: targetDir, stdio: "inherit" });
    } else if (fs.existsSync(requirementsTxt)) {
      console.log("Detected requirements.txt, running pip install...");
      execSync("pip install -r requirements.txt", {
        cwd: targetDir,
        stdio: "inherit",
      });
    } else {
      console.log("No package.json or requirements.txt found. Skipping dependency install.");
    }

    const installed = readJson(INSTALLED_FILE);
    installed.plugins.push({
      name: pluginName,
      source: match,
      folder: repoFolder,
      path: targetDir,
      installed_at: new Date().toISOString(),
    });
    writeJson(INSTALLED_FILE, installed);

    console.log(`Installed plugin: ${pluginName}`);
  });

plugin
  .command("list")
  .description("List installed plugins")
  .action(() => {
    ensureBase();
    const data = readJson(INSTALLED_FILE);

    if (!data.plugins.length) {
      console.log("No plugins installed.");
      return;
    }

    console.log("Installed plugins:");
    data.plugins.forEach((p, i) => {
      console.log(`${i + 1}. ${p.name} -> ${p.source}`);
    });
  });

plugin
  .command("run <name>")
  .description("Run installed plugin")
  .action((name) => {
    ensureBase();
    const data = readJson(INSTALLED_FILE);
    const plugin = data.plugins.find((p) => p.name === name);

    if (!plugin) {
      console.error(`Plugin not found: ${name}`);
      process.exit(1);
    }

    const pkgJsonPath = path.join(plugin.path, "package.json");
    const indexJs = path.join(plugin.path, "index.js");
    const mainPy = path.join(plugin.path, "main.py");

    if (fs.existsSync(pkgJsonPath)) {
      const pkg = fs.readJsonSync(pkgJsonPath);
      if (pkg.scripts && pkg.scripts.start) {
        execSync("npm run start", { cwd: plugin.path, stdio: "inherit" });
        return;
      }
    }

    if (fs.existsSync(indexJs)) {
      execSync("node index.js", { cwd: plugin.path, stdio: "inherit" });
      return;
    }

    if (fs.existsSync(mainPy)) {
      execSync("python main.py", { cwd: plugin.path, stdio: "inherit" });
      return;
    }

    console.error(
      "Could not find a runnable entry point. Expected package.json start script, index.js, or main.py"
    );
    process.exit(1);
  });

program.parse(process.argv);