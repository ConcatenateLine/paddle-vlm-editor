(edited_json) => {
  if (window.editorjsEditor) {
    return window.editorjsEditor.save().then((outputData) => {
      return JSON.stringify(outputData);
    }).catch((error) => {
      console.error('Saving failed: ', error);
      throw error;
    });
  }
  return edited_json;
}
