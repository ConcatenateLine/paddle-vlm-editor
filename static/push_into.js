() => {
  const hidden = document.querySelector('#editorjs_hidden_content textarea');
  if (!hidden) {
    console.warn('[editorjs bridge] hidden textarea (#editorjs_hidden_content) not found in DOM');
    return;
  }
  if (!window.editorjsEditor) {
    console.warn('[editorjs bridge] editorjsEditor not initialized yet');
    return;
  }
  const raw = hidden.value || '';
  console.log('[PUSH_INTO_EDITORJS] Hidden textarea found:', hidden);
  console.log('[PUSH_INTO_EDITORJS] Hidden textarea value length:', raw.length);
  console.log('[PUSH_INTO_EDITORJS] Hidden textarea value preview:', raw.substring(0, 200));
  console.log('[PUSH_INTO_EDITORJS] Hidden textarea value empty:', raw.length === 0);
   
   if (raw.length === 0) {
      console.warn('[PUSH_INTO_EDITORJS] Hidden textarea is empty - nothing to push');
      return;
   }
   
  const editorJsData = window.jsonToEditorJs ? window.jsonToEditorJs(raw) : raw;
  console.log('[PUSH_INTO_EDITORJS] Generated Editor.js data length:', editorJsData.length);
  console.log('[PUSH_INTO_EDITORJS] Generated Editor.js data preview:', editorJsData.substring(0, 200));
  console.log('[PUSH_INTO_EDITORJS] Rendering data into editor');
  
  try {
    const data = typeof editorJsData === 'string' ? JSON.parse(editorJsData) : editorJsData;
    window.editorjsEditor.render(data).then(() => {
      console.log('[PUSH_INTO_EDITORJS] Data rendered successfully');
      if (window.editorjsHistory) {
        window.editorjsHistory.initialize(data);
        console.log('[PUSH_INTO_EDITORJS] Undo history re-initialized with new data');
      }
      const undoBtn = document.getElementById('editorjs-undo-btn');
      const redoBtn = document.getElementById('editorjs-redo-btn');
      if (undoBtn) undoBtn.disabled = true;
      if (redoBtn) redoBtn.disabled = true;
    }).catch((error) => {
      console.error('[PUSH_INTO_EDITORJS] Render failed:', error);
    });
  } catch (e) {
    console.error('[PUSH_INTO_EDITORJS] JSON parse error:', e);
  }
}
