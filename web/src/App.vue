<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import MarkdownIt from 'markdown-it'
import { ElMessage } from 'element-plus'
import {
  ask,
  edit,
  getMessages,
  getReport,
  getSession,
  listSessions,
  startSession,
} from './api'

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

const topic = ref('')
const sessionId = ref('')
const status = ref('idle')
const events = ref([])
const report = ref('')
const error = ref('')
const messages = ref([])
const question = ref('')
const mode = ref('ask')
const sending = ref(false)
const history = ref([])
let timer = null

const renderedReport = computed(() => md.render(report.value || ''))
const renderedMessages = computed(() =>
  messages.value.map((message) => ({
    ...message,
    html: md.render(message.content || ''),
  })),
)
const statusLabel = computed(
  () =>
    ({ idle: '待开始', running: '研究中', ready: '已完成', failed: '失败' }[
      status.value
    ]),
)
const statusType = computed(
  () =>
    ({ idle: 'info', running: 'warning', ready: 'success', failed: 'danger' }[
      status.value
    ]),
)

function detail(err) {
  return err?.response?.data?.detail || err?.message || '未知错误'
}

function resetResults() {
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
  sessionId.value = ''
  events.value = []
  report.value = ''
  messages.value = []
  error.value = ''
  status.value = 'idle'
}

function newSession() {
  resetResults()
  topic.value = ''
}

async function start() {
  const text = topic.value.trim()
  if (!text) {
    ElMessage.warning('请输入研究主题')
    return
  }

  resetResults()
  status.value = 'running'
  try {
    const { data } = await startSession(text)
    sessionId.value = data.session_id
    poll()
  } catch (err) {
    status.value = 'failed'
    error.value = detail(err)
  }
}

async function poll() {
  if (!sessionId.value) return
  try {
    const { data } = await getSession(sessionId.value)
    status.value = data.status
    events.value = data.events || []

    if (data.status === 'running') {
      timer = setTimeout(poll, 2000)
    } else if (data.status === 'ready') {
      await loadReport(sessionId.value)
      await refreshHistory()
    } else if (data.status === 'failed') {
      error.value = data.error || '任务执行失败'
    }
  } catch (err) {
    status.value = 'failed'
    error.value = detail(err)
  }
}

async function loadReport(id) {
  const { data } = await getReport(id)
  report.value = data.content
}

async function loadHistory(id) {
  resetResults()
  sessionId.value = id
  status.value = 'ready'
  try {
    await loadReport(id)
    const { data } = await getSession(id)
    events.value = data.events || []
    topic.value = data.topic || ''

    const { data: historyData } = await getMessages(id)
    messages.value = (historyData.messages || []).map((message) => ({
      role: message.role,
      content: message.content,
    }))
  } catch (err) {
    error.value = detail(err)
  }
}

async function refreshHistory() {
  try {
    const { data } = await listSessions()
    history.value = data.sessions || []
  } catch {
    // 历史列表失败不阻断主流程
  }
}

async function send() {
  const text = question.value.trim()
  if (!text || !sessionId.value || sending.value) return

  messages.value.push({ role: 'user', content: text })
  question.value = ''
  sending.value = true
  try {
    if (mode.value === 'ask') {
      const { data } = await ask(sessionId.value, text)
      messages.value.push({ role: 'assistant', content: data.answer })
    } else {
      const { data } = await edit(sessionId.value, text)
      messages.value.push({ role: 'assistant', content: data.message })
      report.value = data.content
    }
  } catch (err) {
    messages.value.push({
      role: 'assistant',
      content: `请求失败：${detail(err)}`,
    })
  } finally {
    sending.value = false
  }
}

async function copyReport() {
  try {
    await navigator.clipboard.writeText(report.value)
    ElMessage.success('已复制报告')
  } catch {
    ElMessage.error('复制失败')
  }
}

function downloadReport() {
  const blob = new Blob([report.value], {
    type: 'text/markdown;charset=utf-8',
  })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${sessionId.value || 'report'}.md`
  link.click()
  URL.revokeObjectURL(url)
}

onMounted(refreshHistory)
onUnmounted(() => {
  if (timer) clearTimeout(timer)
})
</script>

<template>
  <el-container class="app">
    <el-aside width="300px" class="sidebar">
      <div class="brand">
        <h2>研究报告团队</h2>
        <span class="sub">LangGraph · RAG · Multi-Agent</span>
      </div>

      <el-button class="new-btn" type="primary" plain @click="newSession">
        ＋ 新的研究
      </el-button>

      <div class="history-title">
        <span>历史会话</span>
        <el-button link size="small" @click="refreshHistory">刷新</el-button>
      </div>

      <el-scrollbar class="history-list">
        <div
          v-for="id in history"
          :key="id"
          class="history-item"
          :class="{ active: id === sessionId }"
          @click="loadHistory(id)"
        >
          {{ id }}
        </div>
        <el-empty
          v-if="history.length === 0"
          description="暂无历史"
          :image-size="60"
        />
      </el-scrollbar>
    </el-aside>

    <el-main class="main">
      <section class="topic-bar">
        <el-input
          v-model="topic"
          placeholder="输入研究主题，例如：多智能体系统在软件工程中的应用"
          size="large"
          :disabled="status === 'running'"
          @keyup.enter="start"
        >
          <template #append>
            <el-button
              type="primary"
              :loading="status === 'running'"
              @click="start"
            >
              开始研究
            </el-button>
          </template>
        </el-input>
      </section>

      <el-alert
        v-if="error"
        :title="error"
        type="error"
        show-icon
        closable
        @close="error = ''"
      />

      <section v-if="events.length" class="card">
        <div class="section-head">
          <h3>执行进度</h3>
          <el-tag :type="statusType">{{ statusLabel }}</el-tag>
        </div>
        <el-scrollbar height="160px">
          <div v-for="(event, index) in events" :key="index" class="event-line">
            {{ event }}
          </div>
        </el-scrollbar>
      </section>

      <section v-if="report" class="card">
        <div class="section-head">
          <h3>研究报告</h3>
          <div class="actions">
            <el-button size="small" @click="copyReport">复制</el-button>
            <el-button size="small" @click="downloadReport">下载</el-button>
          </div>
        </div>
        <article class="markdown" v-html="renderedReport"></article>
      </section>

      <section v-if="report" class="card">
        <div class="section-head">
          <h3>继续对话</h3>
          <el-radio-group v-model="mode" size="small">
            <el-radio-button value="ask">提问</el-radio-button>
            <el-radio-button value="edit">修改报告</el-radio-button>
          </el-radio-group>
        </div>

        <div class="messages">
          <div v-if="messages.length === 0" class="hint">
            可以基于报告继续提问，或切换到「修改报告」让写手改写。
          </div>
          <div
            v-for="(message, index) in renderedMessages"
            :key="index"
            class="message"
            :class="message.role"
          >
            <div class="bubble" v-html="message.html"></div>
          </div>
        </div>

        <div class="composer">
          <el-input
            v-model="question"
            :placeholder="
              mode === 'ask'
                ? '问点什么…'
                : '说明要怎么改，例如：把报告精简成三段'
            "
            size="large"
            :disabled="sending"
            @keyup.enter="send"
          />
          <el-button
            type="primary"
            size="large"
            :loading="sending"
            @click="send"
          >
            发送
          </el-button>
        </div>
      </section>

      <el-empty
        v-if="!events.length && !report"
        description="输入主题，开始第一次研究"
      />
    </el-main>
  </el-container>
</template>

<style scoped>
.app {
  height: 100vh;
}

.sidebar {
  background: #fff;
  border-right: 1px solid #e5e7eb;
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.brand h2 {
  margin: 0;
  font-size: 18px;
}

.brand .sub {
  color: #909399;
  font-size: 12px;
}

.new-btn {
  width: 100%;
}

.history-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #606266;
  font-size: 13px;
  margin-top: 8px;
}

.history-list {
  flex: 1;
}

.history-item {
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  color: #606266;
  word-break: break-all;
}

.history-item:hover {
  background: #f2f6fc;
}

.history-item.active {
  background: #ecf5ff;
  color: #409eff;
}

.main {
  padding: 24px 32px;
  width: 100%;
  max-width: 1100px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
}

.topic-bar {
  background: #fff;
  padding: 12px;
  border-radius: 12px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.card {
  background: #fff;
  border-radius: 12px;
  padding: 18px 20px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.section-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.section-head h3 {
  margin: 0;
  font-size: 16px;
}

.actions {
  display: flex;
  gap: 8px;
}

.event-line {
  font-family: Consolas, monospace;
  font-size: 12px;
  color: #606266;
  padding: 3px 0;
  border-bottom: 1px dashed #f0f2f5;
}

.messages {
  max-height: 360px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 12px;
}

.message {
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.bubble {
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 12px;
  background: #f4f4f5;
  font-size: 14px;
  line-height: 1.6;
}

.message.user .bubble {
  background: #409eff;
  color: #fff;
}

.hint {
  color: #909399;
  font-size: 13px;
}

.composer {
  display: flex;
  gap: 10px;
}
</style>
