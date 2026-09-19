#!/usr/bin/env node
/**
 * Write the package.json and package-lock.json of a scratch consumer project
 * that depends on locally built fuse-mojo tarballs, so the project can be
 * installed with `npm ci`.
 *
 * `npm ci` installs exactly what the lockfile records and verifies every
 * package against its integrity hash. `npm install <tarball>` cannot do that:
 * it resolves the dependency tree and writes the lockfile as a side effect of
 * installing, so nothing is pinned before the install runs. Producing the
 * lockfile first, from data we already trust, turns the end-user smoke into a
 * hash-verified install.
 *
 * Usage, from the consumer directory:
 *
 *   node consumer-lockfile.cjs <workspace package-lock.json> <tarball>...
 *
 * Each tarball path is relative to the consumer directory and becomes a
 * `file:` dependency whose integrity is the sha512 of the tarball as built.
 * Every registry dependency the tarballs pull in (koffi and its per-platform
 * binaries) is copied verbatim -- version, resolved URL, integrity -- from the
 * committed workspace lockfile, so the smoke installs the same koffi the
 * differential suite ran against. An optional dependency with no pinned entry
 * (the platform package for another operating system) is left out, exactly as
 * npm leaves it out of node_modules.
 *
 * Node built-ins and `tar` only; npm itself is never invoked.
 */
'use strict'

const crypto = require('node:crypto')
const fs = require('node:fs')
const { execFileSync } = require('node:child_process')

const ROOT_NAME = 'fuse-mojo-smoke'
const ROOT_VERSION = '0.0.0'
// Manifest fields npm records on a lockfile entry for a tarball dependency.
const RECORDED_FIELDS = ['version', 'license', 'os', 'cpu', 'engines', 'dependencies', 'optionalDependencies']

function fail(message) {
  console.error(`consumer-lockfile: ${message}`)
  process.exit(1)
}

function integrityOf(file) {
  return 'sha512-' + crypto.createHash('sha512').update(fs.readFileSync(file)).digest('base64')
}

function manifestOf(tarball) {
  // npm pack always stores the manifest at package/package.json.
  const json = execFileSync('tar', ['-xOzf', tarball, 'package/package.json'], { encoding: 'utf8' })
  return JSON.parse(json)
}

function main(argv) {
  const [workspaceLockfile, ...tarballs] = argv
  if (!workspaceLockfile || tarballs.length === 0) {
    fail('usage: node consumer-lockfile.cjs <workspace package-lock.json> <tarball>...')
  }
  const pinned = JSON.parse(fs.readFileSync(workspaceLockfile, 'utf8')).packages
  if (!pinned) fail(`${workspaceLockfile} has no "packages" section (lockfileVersion 2 or 3 required)`)

  const rootDependencies = {}
  const packages = {}
  const local = new Map() // package name -> manifest, for the tarballs we install directly
  for (const tarball of tarballs) {
    const manifest = manifestOf(tarball)
    if (local.has(manifest.name)) fail(`${manifest.name} is provided by more than one tarball`)
    local.set(manifest.name, manifest)
    const spec = `file:${tarball}`
    rootDependencies[manifest.name] = spec
    const entry = { version: manifest.version, resolved: spec, integrity: integrityOf(tarball) }
    for (const field of RECORDED_FIELDS) {
      if (field !== 'version' && manifest[field] !== undefined) entry[field] = manifest[field]
    }
    packages[`node_modules/${manifest.name}`] = entry
  }

  // Walk the registry closure: every dependency of an installed package that
  // no tarball provides must come, hash and all, from the workspace lockfile.
  const queue = []
  const enqueue = (from, manifest) => {
    for (const name of Object.keys(manifest.dependencies || {})) queue.push({ from, name, optional: false })
    for (const name of Object.keys(manifest.optionalDependencies || {})) queue.push({ from, name, optional: true })
  }
  for (const manifest of local.values()) enqueue(manifest.name, manifest)
  let copied = 0
  while (queue.length > 0) {
    const { from, name, optional } = queue.shift()
    const key = `node_modules/${name}`
    if (local.has(name) || packages[key]) continue
    const entry = pinned[key]
    if (!entry || !entry.resolved || !entry.integrity) {
      if (optional) continue
      fail(`${from} depends on ${name}, which has no pinned entry in ${workspaceLockfile}`)
    }
    packages[key] = entry
    copied += 1
    enqueue(name, entry)
  }

  const sortedPackages = Object.fromEntries(Object.entries(packages).sort(([a], [b]) => a.localeCompare(b)))
  const root = { name: ROOT_NAME, version: ROOT_VERSION, dependencies: rootDependencies }
  const packageJson = { ...root, private: true }
  const lockfile = {
    name: ROOT_NAME,
    version: ROOT_VERSION,
    lockfileVersion: 3,
    requires: true,
    packages: { '': root, ...sortedPackages },
  }
  fs.writeFileSync('package.json', JSON.stringify(packageJson, null, 2) + '\n')
  fs.writeFileSync('package-lock.json', JSON.stringify(lockfile, null, 2) + '\n')
  console.log(`wrote package-lock.json: ${local.size} local tarball(s), ${copied} registry package(s) pinned from ${workspaceLockfile}`)
}

main(process.argv.slice(2))
