import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

export const startSession = (topic) => http.post('/sessions', { topic })
export const getSession = (id) => http.get(`/sessions/${id}`)
export const getMessages = (id) => http.get(`/sessions/${id}/messages`)
export const getReport = (id) => http.get(`/sessions/${id}/report`)
export const ask = (id, question) =>
  http.post(`/sessions/${id}/ask`, { question })
export const edit = (id, instruction) =>
  http.post(`/sessions/${id}/edit`, { instruction })
export const listSessions = () => http.get('/sessions')
