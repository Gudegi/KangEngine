document.addEventListener("DOMContentLoaded", () => {
  const navigation = document.querySelector(".sidebar-tree");
  if (navigation) {
    const defaultOpenSections = new Set(["User Guide", "Integrations"]);
    const topLevelItems = navigation.querySelectorAll(":scope > ul > li");
    for (const item of topLevelItems) {
      const link = item.querySelector(":scope > a");
      if (!defaultOpenSections.has(link?.textContent.trim())) continue;
      const checkbox = item.querySelector(":scope > .toctree-checkbox");
      if (checkbox) {
        checkbox.checked = true;
      }
    }
  }

  const toc = document.querySelector(".toc-tree");
  if (!toc) return;

  const groups = [];
  const candidates = toc.querySelectorAll(":scope > ul > li > ul li");

  for (const item of candidates) {
    const children = Array.from(item.children);
    const nested = children.find((child) => child.tagName === "UL");
    const link = children.find((child) => child.tagName === "A");
    if (!nested || !link) continue;

    item.classList.add("kangengine-toc-fold");
    nested.hidden = true;

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "kangengine-toc-toggle";
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", `Expand ${link.textContent.trim()}`);
    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      toggle.setAttribute(
        "aria-label",
        `${expanded ? "Expand" : "Collapse"} ${link.textContent.trim()}`,
      );
      nested.hidden = expanded;
    });
    item.insertBefore(toggle, nested);
    groups.push({ item, nested, toggle });
  }

  const revealCurrent = () => {
    for (const group of groups) {
      if (!group.item.querySelector(".scroll-current")) continue;
      group.nested.hidden = false;
      group.toggle.setAttribute("aria-expanded", "true");
    }
  };

  revealCurrent();
  new MutationObserver(revealCurrent).observe(toc, {
    subtree: true,
    attributes: true,
    attributeFilter: ["class"],
  });
});
