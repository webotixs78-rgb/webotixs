'use client'

import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import ImageExtension from '@tiptap/extension-image'
import LinkExtension from '@tiptap/extension-link'
import Placeholder from '@tiptap/extension-placeholder'
import {
  Bold,
  Italic,
  Strikethrough,
  Heading1,
  Heading2,
  Heading3,
  List,
  ListOrdered,
  Quote,
  Code,
  Link as LinkIcon,
  Image as ImageIcon,
  Undo,
  Redo,
  Unlink,
} from 'lucide-react'
import { useCallback } from 'react'
import { cn } from '@/lib/utils'

interface RichTextEditorProps {
  content: string
  onChange: (html: string) => void
  placeholder?: string
}

export function RichTextEditor({ content, onChange, placeholder = 'Write your article story...' }: RichTextEditorProps) {
  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: {
          levels: [1, 2, 3],
        },
      }),
      ImageExtension.configure({
        inline: true,
        allowBase64: true,
      }),
      LinkExtension.configure({
        openOnClick: false,
        autolink: true,
      }),
      Placeholder.configure({
        placeholder,
      }),
    ],
    content,
    editorProps: {
      attributes: {
        class:
          'prose prose-invert max-w-none min-h-[220px] p-4 bg-[#050816] border border-[#273449] rounded-b-2xl text-white text-sm focus:outline-none focus:border-blue-500/50 leading-relaxed font-sans',
      },
    },
    onUpdate: ({ editor }) => {
      onChange(editor.getHTML())
    },
  })

  const setLink = useCallback(() => {
    if (!editor) return
    const previousUrl = editor.getAttributes('link').href
    const url = window.prompt('Enter URL:', previousUrl)

    // cancelled
    if (url === null) return

    // empty
    if (url === '') {
      editor.chain().focus().extendMarkRange('link').unsetLink().run()
      return
    }

    // update link
    editor.chain().focus().extendMarkRange('link').setLink({ href: url }).run()
  }, [editor])

  const addImage = useCallback(() => {
    if (!editor) return
    const url = window.prompt('Enter Image URL or select from Media Library:')
    if (url) {
      editor.chain().focus().setImage({ src: url }).run()
    }
  }, [editor])

  if (!editor) {
    return (
      <div className="w-full h-64 bg-[#050816] border border-[#273449] rounded-2xl flex items-center justify-center text-[#94A3B8] text-xs">
        Loading Tiptap Editor...
      </div>
    )
  }

  return (
    <div className="border border-[#273449] rounded-2xl overflow-hidden bg-[#0D1224]">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-1 p-2 border-b border-[#273449] bg-[#0A0E1F]">
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleBold().run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('bold') ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Bold (Ctrl+B)"
        >
          <Bold size={14} />
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleItalic().run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('italic') ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Italic (Ctrl+I)"
        >
          <Italic size={14} />
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleStrike().run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('strike') ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Strikethrough"
        >
          <Strikethrough size={14} />
        </button>

        <div className="w-px h-5 bg-[#273449] mx-1" />

        <button
          type="button"
          onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('heading', { level: 1 }) ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Heading 1"
        >
          <Heading1 size={14} />
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('heading', { level: 2 }) ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Heading 2"
        >
          <Heading2 size={14} />
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('heading', { level: 3 }) ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Heading 3"
        >
          <Heading3 size={14} />
        </button>

        <div className="w-px h-5 bg-[#273449] mx-1" />

        <button
          type="button"
          onClick={() => editor.chain().focus().toggleBulletList().run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('bulletList') ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Bullet List"
        >
          <List size={14} />
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleOrderedList().run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('orderedList') ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Numbered List"
        >
          <ListOrdered size={14} />
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleBlockquote().run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('blockquote') ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Quote"
        >
          <Quote size={14} />
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleCodeBlock().run()}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('codeBlock') ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Code Block"
        >
          <Code size={14} />
        </button>

        <div className="w-px h-5 bg-[#273449] mx-1" />

        <button
          type="button"
          onClick={setLink}
          className={cn(
            'p-2 rounded-lg text-xs font-semibold transition-colors',
            editor.isActive('link') ? 'bg-blue-600 text-white' : 'text-[#94A3B8] hover:text-white hover:bg-white/5'
          )}
          title="Insert Link"
        >
          <LinkIcon size={14} />
        </button>
        {editor.isActive('link') && (
          <button
            type="button"
            onClick={() => editor.chain().focus().unsetLink().run()}
            className="p-2 rounded-lg text-xs font-semibold text-red-400 hover:bg-red-500/10 transition-colors"
            title="Remove Link"
          >
            <Unlink size={14} />
          </button>
        )}
        <button
          type="button"
          onClick={addImage}
          className="p-2 rounded-lg text-xs font-semibold text-[#94A3B8] hover:text-white hover:bg-white/5 transition-colors"
          title="Insert Image"
        >
          <ImageIcon size={14} />
        </button>

        <div className="w-px h-5 bg-[#273449] mx-1" />

        <button
          type="button"
          onClick={() => editor.chain().focus().undo().run()}
          disabled={!editor.can().undo()}
          className="p-2 rounded-lg text-xs font-semibold text-[#94A3B8] hover:text-white hover:bg-white/5 disabled:opacity-30 transition-colors"
          title="Undo"
        >
          <Undo size={14} />
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().redo().run()}
          disabled={!editor.can().redo()}
          className="p-2 rounded-lg text-xs font-semibold text-[#94A3B8] hover:text-white hover:bg-white/5 disabled:opacity-30 transition-colors"
          title="Redo"
        >
          <Redo size={14} />
        </button>
      </div>

      {/* Editor Content Body */}
      <EditorContent editor={editor} />
    </div>
  )
}
