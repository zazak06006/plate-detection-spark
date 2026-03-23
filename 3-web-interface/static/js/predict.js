// Predict page — drag & drop + preview + loader
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview-container");
const previewImg = document.getElementById("preview-img");
const previewName = document.getElementById("preview-name");
const form = document.getElementById("predict-form");
const btnText = document.getElementById("btn-text");
const btnLoader = document.getElementById("btn-loader");

if (dropZone) {
    // Click to open file dialog
    dropZone.addEventListener("click", e => {
        if (e.target !== fileInput) fileInput.click();
    });

    // Drag events
    dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    // File input change
    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        // Assign to input
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        // Preview
        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            previewName.textContent = file.name;
            preview.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    // Form submit loader
    form.addEventListener("submit", () => {
        if (fileInput.files.length > 0) {
            btnText.classList.add("hidden");
            btnLoader.classList.remove("hidden");
        }
    });
}
