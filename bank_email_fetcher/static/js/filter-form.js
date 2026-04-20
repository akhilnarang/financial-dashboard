document.querySelectorAll("form.filter-bar").forEach((form) => {
    form.addEventListener("formdata", (event) => {
        for (const [name, value] of Array.from(event.formData.entries())) {
            if (value === "") event.formData.delete(name);
        }
    });
});
