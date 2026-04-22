if (typeof File === 'undefined') {
  globalThis.File = class File extends Blob {
    constructor(bits = [], name = '', options = {}) {
      super(bits, options);
      this.name = String(name);
      this.lastModified = options.lastModified ?? Date.now();
      this.webkitRelativePath = '';
    }
  };
}
