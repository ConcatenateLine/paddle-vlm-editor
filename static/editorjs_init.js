() => {
  function initEditorJs() {
    const target = document.getElementById('editorjs');
    console.log('[EDITORJS_INIT] Looking for editorjs element:', target);
    console.log('[EDITORJS_INIT] EditorJS available:', typeof EditorJS !== 'undefined');
    console.log('[EDITORJS_INIT] SimpleImage available:', typeof window.SimpleImage !== 'undefined');
    if (!target || typeof EditorJS === 'undefined' || typeof window.SimpleImage === 'undefined') {
      console.log('[EDITORJS_INIT] Retrying in 200ms...');
      setTimeout(initEditorJs, 200);
      return;
    }
    
    if (window.editorjsEditor) {
      console.log('[EDITORJS_INIT] Editor.js already initialized');
      return;
    }
 
    console.log('[EDITORJS_INIT] Initializing Editor.js');
    console.log('[EDITORJS_INIT] Available plugins:', {
      Header: typeof window.Header,
      List: typeof window.List,
      CodeTool: typeof window.CodeTool,
      Quote: typeof window.Quote,
      Delimiter: typeof window.Delimiter,
      Bold: typeof window.Bold,
      Italic: typeof window.Italic,
      Underline: typeof window.Underline,
      LinkTool: typeof window.LinkTool,
      Marker: typeof window.Marker,
      Table: typeof window.Table,
      EditorjsTable: typeof window.EditorjsTable,
      ImageTool: typeof window.ImageTool,
      SimpleImage: typeof window.SimpleImage,
      Checklist: typeof window.Checklist,
      Warning: typeof window.Warning,
      Embed: typeof window.Embed,
      AttachesTool: typeof window.AttachesTool,
      RawTool: typeof window.RawTool
    });

    const tools = {};
    if (typeof window.Header !== 'undefined') {
      tools.header = {
        class: window.Header,
        config: {
          levels: [1, 2, 3],
          defaultLevel: 2
        },
        tunes: ['alignment']
      };
    }
    if (typeof window.List !== 'undefined') {
      tools.list = {
        class: window.List,
        inlineToolbar: true,
        tunes: ['alignment']
      };
    }
    if (typeof window.CodeTool !== 'undefined') {
      tools.code = window.CodeTool;
    }
    if (typeof window.Quote !== 'undefined') {
      tools.quote = {
        class: window.Quote,
        inlineToolbar: true,
        tunes: ['alignment']
      };
    }
    if (typeof window.Delimiter !== 'undefined') {
      tools.delimiter = window.Delimiter;
    }
    if (typeof window.Bold !== 'undefined') {
      tools.bold = window.Bold;
    }
    if (typeof window.Italic !== 'undefined') {
      tools.italic = window.Italic;
    }
    if (typeof window.Underline !== 'undefined') {
      tools.underline = window.Underline;
    }
    if (typeof window.LinkTool !== 'undefined') {
      tools.link = window.LinkTool;
    }
    if (typeof window.Marker !== 'undefined') {
      tools.marker = window.Marker;
    }
    if (typeof window.Table !== 'undefined') {
      tools.table = {
        class: window.Table,
        inlineToolbar: true
      };
    } else if (typeof window.EditorjsTable !== 'undefined') {
      tools.table = {
        class: window.EditorjsTable,
        inlineToolbar: true
      };
    }
    if (typeof window.ImageTool !== 'undefined') {
      tools.image = {
        class: window.ImageTool,
        inlineToolbar: true
      };
    } else if (typeof window.SimpleImage !== 'undefined') {
      tools['simple-image'] = {
        class: window.SimpleImage,
        inlineToolbar: true
      };
    }
    if (typeof window.Checklist !== 'undefined') {
      tools.checklist = {
        class: window.Checklist,
        inlineToolbar: true
      };
    }
    if (typeof window.Warning !== 'undefined') {
      tools.warning = window.Warning;
    }
    if (typeof window.Embed !== 'undefined') {
      tools.embed = window.Embed;
    }
    if (typeof window.AttachesTool !== 'undefined') {
      tools.attaches = window.AttachesTool;
    }
    if (typeof window.RawTool !== 'undefined') {
      tools.raw = window.RawTool;
    }
    if (typeof window.AlignmentTune !== 'undefined') {
      tools.alignment = window.AlignmentTune;
    }

    window.editorjsEditor = new EditorJS({
      holder: 'editorjs',
      placeholder: 'Run a pipeline or pick a file from the workspace history to edit its output...',
      tools: tools,
      data: {
        blocks: []
      },
      onChange: () => {
        if (window.editorjsHistory && !window.editorjsHistory._restoring) {
          window.editorjsHistory._debouncedSave();
        }
        window.editorjsEditor.save().then((outputData) => {
          const hidden = document.querySelector('#editorjs_hidden_content textarea');
          if (hidden) {
            hidden.value = JSON.stringify(outputData);
            hidden.dispatchEvent(new Event('input', { bubbles: true }));
          }
        }).catch((error) => {
          console.error('Saving failed: ', error);
        });
      }
    });
    class EditorHistory {
      constructor(editor, maxStack) {
        this.editor = editor;
        this.undoStack = [];
        this.redoStack = [];
        this.maxStack = maxStack || 100;
        this._restoring = false;
        this._debounceTimer = null;
        this.undoBtn = document.getElementById('editorjs-undo-btn');
        this.redoBtn = document.getElementById('editorjs-redo-btn');
        this._bindButtons();
        this._bindKeyboard();
      }
      _bindButtons() {
        const self = this;
        if (this.undoBtn) {
          this.undoBtn.addEventListener('click', () => { self.undo(); });
        }
        if (this.redoBtn) {
          this.redoBtn.addEventListener('click', () => { self.redo(); });
        }
      }
      _bindKeyboard() {
        const self = this;
        document.addEventListener('keydown', (e) => {
          const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
          const mod = isMac ? e.metaKey : e.ctrlKey;
          if (mod && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            self.undo();
          } else if (mod && e.key === 'z' && e.shiftKey) {
            e.preventDefault();
            self.redo();
          } else if (mod && e.key === 'y') {
            e.preventDefault();
            self.redo();
          }
        });
      }
      _updateButtons() {
        if (this.undoBtn) this.undoBtn.disabled = this.undoStack.length === 0;
        if (this.redoBtn) this.redoBtn.disabled = this.redoStack.length === 0;
      }
      _debouncedSave() {
        if (this._restoring) return;
        clearTimeout(this._debounceTimer);
        const self = this;
        this._debounceTimer = setTimeout(() => {
          self._snapshot();
        }, 300);
      }
      async _snapshot() {
        if (this._restoring) return;
        try {
          const data = await this.editor.save();
          const last = this.undoStack[this.undoStack.length - 1];
          if (last && JSON.stringify(last) === JSON.stringify(data)) return;
          this.undoStack.push(data);
          if (this.undoStack.length > this.maxStack) {
            this.undoStack.shift();
          }
          this.redoStack = [];
          this._updateButtons();
        } catch (err) {
          console.error('[EditorHistory] snapshot failed:', err);
        }
      }
      async undo() {
        if (this.undoStack.length === 0) return;
        this._restoring = true;
        try {
          const current = await this.editor.save();
          this.redoStack.push(current);
          const prev = this.undoStack.pop();
          await this.editor.render(prev);
        } catch (err) {
          console.error('[EditorHistory] undo failed:', err);
        }
        this._restoring = false;
        this._updateButtons();
        this._syncHidden();
      }
      async redo() {
        if (this.redoStack.length === 0) return;
        this._restoring = true;
        try {
          const current = await this.editor.save();
          this.undoStack.push(current);
          const next = this.redoStack.pop();
          await this.editor.render(next);
        } catch (err) {
          console.error('[EditorHistory] redo failed:', err);
        }
        this._restoring = false;
        this._updateButtons();
        this._syncHidden();
      }
      async initialize(data) {
        this.undoStack = [];
        this.redoStack = [];
        this._restoring = false;
        try {
          const current = await this.editor.save();
          this.undoStack.push(current);
          this._updateButtons();
        } catch (err) {
          console.error('[EditorHistory] initialize failed:', err);
        }
      }
      _syncHidden() {
        this.editor.save().then((outputData) => {
          const hidden = document.querySelector('#editorjs_hidden_content textarea');
          if (hidden) {
            hidden.value = JSON.stringify(outputData);
            hidden.dispatchEvent(new Event('input', { bubbles: true }));
          }
        });
      }
    }
    window.editorjsHistory = new EditorHistory(window.editorjsEditor, 100);
    console.log('[EDITORJS_INIT] Custom undo/redo manager enabled');
    console.log('[EDITORJS_INIT] Editor.js initialized successfully');
  }
  setTimeout(initEditorJs, 300);
}
